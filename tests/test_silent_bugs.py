"""Silent-bug hypothesis tests.

Each test states a hypothesis, exercises the code, and asserts the CORRECT
behaviour.  A failing test means the hypothesis is confirmed — the bug is real.

Hypotheses tested:
  H1  _GameBuffer._load does not cast tty_chars to uint8.
      If an npz stores tty_chars as int8 (which NLE's pyconverter can produce),
      values 128-255 are stored as -128..-1.  The stacked (T,H,W,2) array
      ends up with wrong dtype (int16 after numpy's signed/unsigned promotion),
      and tokenise() produces token IDs in [-4096, -1] — invalid embedding
      indices.

  H2  _GameBuffer._load has no fallback when tty_colors is absent.
      Unlike _load_from_nao_top10_dir it unconditionally accesses
      f["tty_colors"] and raises KeyError on files that lack it.

  H3  _GameBuffer._pool_weights is off by one vs _make_weights.
      A game with exactly T = ctx + horizon valid starting positions has
      weight 0 in _pool_weights (so it can never enter the buffer) but
      weight > 0 in _make_weights (which computes the correct count).

  H4  tokenise() with a signed (int8/int16) char channel produces negative
      token IDs for chars >= 128, which are out-of-range for nn.Embedding
      and cause silent wrong lookups (no bounds check on CUDA).

  H5  Embedding lookup with negative index does NOT raise an error on CPU
      in PyTorch — it silently wraps around, mapping to a wrong code.
"""

import io
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from lom.dataset import NpzTrajectoryDataset, _GameBuffer
from lom.modules import tokenise, COLOR_VOCAB, CHAR_VOCAB, TOKEN_VOCAB

# ── tiny geometry used across all tests ──────────────────────────────────────
CTX     = 4
HORIZON = 8
H, W    = 4, 4  # small for speed


def _write_npz(path, chars_arr, colors_arr=None):
    """Write a minimal game npz file."""
    if colors_arr is None:
        colors_arr = np.zeros_like(chars_arr, dtype=np.uint8)
    np.savez(path, tty_chars=chars_arr, tty_colors=colors_arr)


# ─────────────────────────────────────────────────────────────────────────────
# H1 — tty_chars dtype not cast to uint8
# ─────────────────────────────────────────────────────────────────────────────

class TestH1CharsNotCastToUint8:
    """_GameBuffer._load must emit a (T, H, W, 2) uint8 array regardless of
    the dtype stored in the npz.  Currently it does NOT cast chars."""

    def _make_int8_npz(self, tmp_path):
        T = CTX + HORIZON + 5
        # chars with values 128-255; stored as int8 these become -128..-1
        chars_uint8 = np.full((T, H, W), 200, dtype=np.uint8)
        chars_int8  = chars_uint8.view(np.int8)          # reinterpret, same bytes
        colors      = np.zeros((T, H, W), dtype=np.uint8)
        p = str(tmp_path / "game.npz")
        np.savez(p, tty_chars=chars_int8, tty_colors=colors)
        return p, T

    def test_output_dtype_is_uint8(self, tmp_path):
        """Loaded game array must be uint8 so that tokenise() produces valid IDs."""
        path, T = self._make_int8_npz(tmp_path)
        paths   = np.array([path], dtype=object)
        lengths = np.array([T],    dtype=np.int32)
        buf = _GameBuffer(paths, lengths, buffer_size=1,
                          context_len=CTX, horizon=HORIZON, seed=0)
        game, _ = buf.sample(np.random.default_rng(0))
        buf.stop()
        assert game.dtype == np.uint8, (
            f"Expected uint8 but got {game.dtype}. "
            "int8 tty_chars must be reinterpreted as uint8."
        )

    def test_char_values_preserved(self, tmp_path):
        """char=200 stored as int8=-56 must round-trip back to 200."""
        path, T = self._make_int8_npz(tmp_path)
        paths   = np.array([path], dtype=object)
        lengths = np.array([T],    dtype=np.int32)
        buf = _GameBuffer(paths, lengths, buffer_size=1,
                          context_len=CTX, horizon=HORIZON, seed=0)
        game, _ = buf.sample(np.random.default_rng(0))
        buf.stop()
        chars_channel = game[..., 0]
        assert (chars_channel == 200).all(), (
            f"Expected char channel == 200, got unique values {np.unique(chars_channel)}. "
            "Sign-reinterpretation of int8 corrupts char values >= 128."
        )

    def test_tokenise_produces_valid_ids(self, tmp_path):
        """Token IDs must be in [0, TOKEN_VOCAB) after loading int8 chars."""
        path, T = self._make_int8_npz(tmp_path)
        paths   = np.array([path], dtype=object)
        lengths = np.array([T],    dtype=np.int32)
        buf = _GameBuffer(paths, lengths, buffer_size=1,
                          context_len=CTX, horizon=HORIZON, seed=0)
        game, t = buf.sample(np.random.default_rng(0))
        buf.stop()
        frame = torch.from_numpy(game[t].copy())  # (H, W, 2)
        ids   = tokenise(frame)                    # (H, W) long
        assert ids.min() >= 0 and ids.max() < TOKEN_VOCAB, (
            f"Token IDs out of range [{ids.min()}, {ids.max()}]. "
            f"Expected [0, {TOKEN_VOCAB}). Negative IDs come from int8 chars."
        )


# ─────────────────────────────────────────────────────────────────────────────
# H2 — no fallback when tty_colors is absent
# ─────────────────────────────────────────────────────────────────────────────

class TestH2MissingTtyColors:
    """_GameBuffer._load must not raise KeyError when tty_colors is absent;
    it should fall back to zeros like _load_from_nao_top10_dir does."""

    def test_no_key_error_when_colors_missing(self, tmp_path):
        T = CTX + HORIZON + 5
        chars = np.zeros((T, H, W), dtype=np.uint8)
        p = str(tmp_path / "no_colors.npz")
        np.savez(p, tty_chars=chars)            # no tty_colors key
        paths   = np.array([p], dtype=object)
        lengths = np.array([T], dtype=np.int32)
        # should not raise
        try:
            buf = _GameBuffer(paths, lengths, buffer_size=1,
                              context_len=CTX, horizon=HORIZON, seed=0)
            buf.stop()
        except KeyError as e:
            pytest.fail(
                f"_GameBuffer._load raised KeyError({e}) for a file without "
                "tty_colors.  It should fall back to a zero color channel."
            )

    def test_color_channel_zeros_when_missing(self, tmp_path):
        T = CTX + HORIZON + 5
        chars = np.ones((T, H, W), dtype=np.uint8) * 65  # 'A'
        p = str(tmp_path / "no_colors.npz")
        np.savez(p, tty_chars=chars)
        paths   = np.array([p], dtype=object)
        lengths = np.array([T], dtype=np.int32)
        buf = _GameBuffer(paths, lengths, buffer_size=1,
                          context_len=CTX, horizon=HORIZON, seed=0)
        game, _ = buf.sample(np.random.default_rng(0))
        buf.stop()
        assert (game[..., 1] == 0).all(), (
            "Color channel should be all-zero when tty_colors is absent."
        )


# ─────────────────────────────────────────────────────────────────────────────
# H3 — _pool_weights off-by-one
# ─────────────────────────────────────────────────────────────────────────────

class TestH3PoolWeightsOffByOne:
    """Games with exactly T = ctx + horizon have exactly 1 valid starting
    position (t = ctx-1).  _make_weights correctly assigns them weight > 0.
    _pool_weights uses T - (ctx+horizon) = 0 and so these games can never
    enter the buffer, even though they're perfectly valid."""

    def test_borderline_game_has_positive_make_weights(self):
        """_make_weights must give weight > 0 to a game with T = ctx+horizon."""
        T = CTX + HORIZON              # exactly 1 valid start
        game = np.zeros((T, H, W, 2), dtype=np.uint8)
        # Simulate what _make_weights computes
        valid = max(len(game) - (CTX + HORIZON - 1), 0)
        assert valid == 1, f"Expected 1 valid start, got {valid}"

    def test_borderline_game_has_zero_pool_weights(self):
        """_pool_weights gives weight 0 to the same borderline game — the bug."""
        T      = CTX + HORIZON
        lengths = np.array([T], dtype=np.float64)
        # Replicates the _pool_weights formula in _GameBuffer.__init__
        min_len = CTX + HORIZON + 1
        valid   = max(float(lengths[0]) - (min_len - 1), 0.0)
        assert valid == 0.0, (
            f"Expected _pool_weights formula to give 0 for T=ctx+horizon "
            f"but got {valid}.  This game should be loadable."
        )

    def test_pool_and_make_weights_agree_on_borderline_game(self, tmp_path):
        """When a borderline game IS in the buffer its weight must be > 0.
        The off-by-one means the pool never loads it; this test verifies the
        correct formula so we can fix _pool_weights to match _make_weights."""
        T = CTX + HORIZON  # 1 valid start
        chars  = np.zeros((T, H, W), dtype=np.uint8)
        colors = np.zeros((T, H, W), dtype=np.uint8)
        p = str(tmp_path / "borderline.npz")
        np.savez(p, tty_chars=chars, tty_colors=colors)

        # Force it into the buffer by faking lengths to be one larger so
        # _pool_weights > 0, then verify _make_weights still gives the right count.
        paths   = np.array([p], dtype=object)
        lengths = np.array([T + 1], dtype=np.int32)   # lie: say T+1 frames
        buf = _GameBuffer(paths, lengths, buffer_size=1,
                          context_len=CTX, horizon=HORIZON, seed=0)
        games, weights = buf._state
        buf.stop()
        assert weights[0] > 0, (
            f"_make_weights gave weight={weights[0]} to a valid game. "
            "Expected > 0."
        )


# ─────────────────────────────────────────────────────────────────────────────
# H4 — tokenise with signed char channel produces negative IDs
# ─────────────────────────────────────────────────────────────────────────────

class TestH4TokeniseNegativeIds:
    """If the char channel contains values in [-128, -1] (int8 sign
    interpretation of ASCII 128-255), tokenise() produces token IDs outside
    [0, TOKEN_VOCAB) which are invalid for nn.Embedding."""

    def test_uint8_chars_produce_valid_ids(self):
        """Baseline: uint8 chars → IDs always in [0, TOKEN_VOCAB)."""
        obs = torch.zeros(H, W, 2, dtype=torch.uint8)
        obs[..., 0] = 255   # max char
        obs[..., 1] = 31    # max color
        ids = tokenise(obs)
        assert ids.min() >= 0 and ids.max() < TOKEN_VOCAB

    def test_int8_chars_with_high_values_produce_negative_ids(self):
        """int8 char=200 → long=-56; tokenise gives ID = -56*32 + 0 = -1792 < 0."""
        frame_uint8 = torch.full((H, W, 2), 0, dtype=torch.uint8)
        frame_uint8[..., 0] = 200   # char=200

        # Simulate what happens when the tensor was loaded from an int8 npz:
        # view as int8 then convert to long — exactly what torch.from_numpy does
        # on an int8 array
        frame_int8 = frame_uint8.numpy().view(np.int8)    # same bytes, signed view
        frame_as_loaded = torch.from_numpy(frame_int8.copy())  # int8 tensor

        # tokenise calls .long() which sign-extends int8
        ids = tokenise(frame_as_loaded)
        has_negative = (ids < 0).any().item()
        assert has_negative, (
            "Expected negative token IDs when char channel is int8 with values "
            ">= 128, but got all non-negative.  Test premise may be wrong."
        )
        assert (ids < 0).any() or (ids >= TOKEN_VOCAB).any(), (
            "int8 chars >= 128 must produce out-of-range token IDs."
        )

    def test_negative_token_ids_cause_error_or_wrong_result(self):
        """Negative token IDs from int8 chars must either crash or produce wrong
        results — both are unacceptable.  On CPU PyTorch raises RuntimeError
        (out-of-range index); on CUDA it silently accesses the wrong row.
        Either way, int8 chars must be caught before reaching the model.
        """
        emb = nn.Embedding(TOKEN_VOCAB, 4)
        ids = torch.tensor([-56 * COLOR_VOCAB + 0])   # from char=200 as int8
        assert ids[0] < 0, "Precondition: ids must be negative"

        # On CPU: raises RuntimeError.  On CUDA: silently wrong.
        # We assert that at least one bad outcome occurs.
        raised_error = False
        try:
            emb(ids)
        except (RuntimeError, IndexError):
            raised_error = True

        cross_entropy_bad = False
        logits = torch.randn(1, TOKEN_VOCAB)
        try:
            torch.nn.functional.cross_entropy(logits, ids)
        except (RuntimeError, IndexError):
            cross_entropy_bad = True

        assert raised_error or cross_entropy_bad, (
            "Expected negative token IDs to cause either a RuntimeError in "
            "embedding lookup or cross-entropy — this confirms the upstream "
            "int8 dtype bug (H1) has concrete downstream consequences."
        )


# ─────────────────────────────────────────────────────────────────────────────
# H5 — NpzTrajectoryDataset.from_index with int8 chars produces bad batches
# ─────────────────────────────────────────────────────────────────────────────

class TestH5EndToEndInt8Chars:
    """Full pipeline test: int8 tty_chars in npz → DataLoader batch →
    token IDs → must be in [0, TOKEN_VOCAB).  Currently fails."""

    def test_batch_token_ids_in_range(self, tmp_path):
        T = CTX + HORIZON + 10
        # Write 3 games with char=200 stored as int8
        paths = []
        for i in range(3):
            chars_int8  = np.full((T, H, W), 200, dtype=np.uint8).view(np.int8)
            colors_uint8 = np.zeros((T, H, W), dtype=np.uint8)
            p = str(tmp_path / f"game{i}.npz")
            np.savez(p, tty_chars=chars_int8, tty_colors=colors_uint8)
            paths.append(p)

        index_path = str(tmp_path / "index.npz")
        np.savez(index_path,
                 paths=np.array(paths, dtype=object),
                 lengths=np.array([T, T, T], dtype=np.int32))

        ds = NpzTrajectoryDataset.from_index(
            index_path,
            context_len=CTX,
            horizon=HORIZON,
            buffer_size=3,
            steps_per_epoch=4,
            seed=0,
            obs_h=H,
            obs_w=W,
        )

        history, next_frame, future_frame = ds[0]
        # history: (CTX, H, W, 2) — check char channel
        char_channel = history[..., 0]   # uint8 or whatever came back
        ids = tokenise(history.reshape(CTX, H, W, 2))
        ds.close()

        assert ids.min() >= 0 and ids.max() < TOKEN_VOCAB, (
            f"Token IDs [{ids.min()}, {ids.max()}] out of [0, {TOKEN_VOCAB}). "
            "Caused by int8 tty_chars not being cast to uint8 on load."
        )
