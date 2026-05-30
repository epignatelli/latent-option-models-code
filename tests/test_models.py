import pytest
import torch
from lom.modules import DynamicsModel, LatentActionModel
from lom.modules import STTEncoder, JEPAEncoder, EMAEncoder

from conftest import BATCH, CONTEXT, D_MODEL, HORIZON, LATENT_DIM, N_HEADS, N_LAYERS, OBS_H, OBS_W, S, VOCAB

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flex_attention backward requires CUDA"
)


def make_lam(in_dim=D_MODEL):
    return LatentActionModel(in_dim=in_dim, latent_dim=LATENT_DIM, num_options=32)


def make_dynamics(option_dim=None, predict_sequence=False):
    return DynamicsModel(
        vocab_size=VOCAB,
        obs_h=OBS_H,
        obs_w=OBS_W,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        context_length=CONTEXT,
        latent_dim=LATENT_DIM,
        option_dim=option_dim,
        predict_sequence=predict_sequence,
        horizon=HORIZON if predict_sequence else 1,
    )


def _screen(*leading):
    """Random (char, color) screen tensor: (*leading, H, W, 2) uint8."""
    chars  = torch.randint(0, 256, (*leading, OBS_H, OBS_W, 1), dtype=torch.uint8)
    colors = torch.randint(0, 32,  (*leading, OBS_H, OBS_W, 1), dtype=torch.uint8)
    return torch.cat([chars, colors], dim=-1)


def history():
    return _screen(BATCH, CONTEXT)


def single_frame():
    return _screen(BATCH)


def frame_sequence(k):
    return _screen(BATCH, k)


# --------------------------------------------------------------------------- #
# --- LatentActionModel (VQ bottleneck) ------------------------------------- #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def test_lam_output_shapes():
    lam = make_lam()
    x = torch.randn(BATCH, D_MODEL)
    z_q, _, indices = lam(x)
    assert z_q.shape == (BATCH, LATENT_DIM)
    assert indices.shape == (BATCH,)


@needs_cuda
def test_lam_backward():
    lam = make_lam().cuda()
    _, loss_dict, _ = lam(torch.randn(BATCH, D_MODEL).cuda())
    loss_dict["vq_loss"].backward()


# --------------------------------------------------------------------------- #
# --- Encoder classes ------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def make_encoder(cls, horizon=1, condition_dim=None):
    kwargs = dict(
        vocab_size=VOCAB, obs_h=OBS_H, obs_w=OBS_W,
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        context_length=CONTEXT, horizon=horizon, condition_dim=condition_dim,
    )
    if cls is JEPAEncoder:
        kwargs["latent_dim"] = LATENT_DIM
    return cls(**kwargs)


@torch.no_grad()
def test_bidirectional_encoder_shape():
    enc = make_encoder(STTEncoder)
    out = enc(history(), single_frame())
    assert out.shape == (BATCH, D_MODEL)


@torch.no_grad()
def test_bidirectional_encoder_sequence():
    enc = make_encoder(STTEncoder, horizon=HORIZON)
    out = enc(history(), frame_sequence(HORIZON))
    assert out.shape == (BATCH, D_MODEL)


@torch.no_grad()
def test_bidirectional_encoder_with_condition():
    enc = make_encoder(STTEncoder, condition_dim=LATENT_DIM)
    out = enc(history(), single_frame(), condition=torch.randn(BATCH, LATENT_DIM))
    assert out.shape == (BATCH, D_MODEL)


@torch.no_grad()
def test_independent_encoder_shape():
    enc = make_encoder(JEPAEncoder)
    out = enc(history(), single_frame())
    assert out.shape == (BATCH, 2 * D_MODEL)


@torch.no_grad()
def test_independent_encoder_sequence():
    enc = make_encoder(JEPAEncoder, horizon=HORIZON)
    out = enc(history(), frame_sequence(HORIZON))
    assert out.shape == (BATCH, 2 * D_MODEL)


@torch.no_grad()
def test_independent_encoder_with_condition():
    enc = make_encoder(JEPAEncoder, condition_dim=LATENT_DIM)
    out = enc(history(), single_frame(), condition=torch.randn(BATCH, LATENT_DIM))
    assert out.shape == (BATCH, 2 * D_MODEL)


def test_lam_serialise_roundtrip(tmp_path):
    lam = make_lam().eval()
    x = torch.randn(BATCH, D_MODEL)
    with torch.no_grad():
        z_before, _, _ = lam(x)
    lam.save(str(tmp_path / "lam.pt"))
    lam2 = LatentActionModel.load(str(tmp_path / "lam.pt")).eval()
    with torch.no_grad():
        z_after, _, _ = lam2(x)
    assert torch.allclose(z_before, z_after)


def test_dynamics_serialise_roundtrip(tmp_path):
    dyn = make_dynamics().eval()
    hist, act = history(), action()
    with torch.no_grad():
        logits_before = dyn(hist, act)
    dyn.save(str(tmp_path / "dyn.pt"))
    dyn2 = DynamicsModel.load(str(tmp_path / "dyn.pt")).eval()
    with torch.no_grad():
        logits_after = dyn2(hist, act)
    assert torch.allclose(logits_before, logits_after)


# --------------------------------------------------------------------------- #
# --- DynamicsModel --------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def action():
    return torch.randn(BATCH, LATENT_DIM)


@torch.no_grad()
def test_dynamics_single_frame():
    dyn = make_dynamics()
    logits = dyn(history(), action())
    assert logits.shape == (BATCH, S, VOCAB)


@torch.no_grad()
def test_dynamics_sequence_teacher():
    dyn = make_dynamics(predict_sequence=True)
    teacher = frame_sequence(HORIZON)
    logits = dyn(history(), action(), horizon=HORIZON, teacher_frames=teacher)
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)


@torch.no_grad()
def test_dynamics_sequence_autoregressive():
    dyn = make_dynamics(predict_sequence=True)
    logits = dyn(history(), action(), horizon=HORIZON)
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)


@torch.no_grad()
def test_dynamics_with_option_code():
    dyn = make_dynamics(option_dim=LATENT_DIM)
    option_code = torch.randn(BATCH, LATENT_DIM)
    logits = dyn(history(), action(), option_code=option_code)
    assert logits.shape == (BATCH, S, VOCAB)


@needs_cuda
def test_dynamics_backward():
    dyn = make_dynamics().cuda()
    logits = dyn(history().cuda(), action().cuda())
    target = torch.randint(0, VOCAB, (BATCH, S), device="cuda")
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()


# --------------------------------------------------------------------------- #
# --- Patch embedding in models --------------------------------------------- #
# --------------------------------------------------------------------------- #

PATCH = 2  # OBS_H=OBS_W=4, so 4//2=2 tokens per dim → 4 tokens/frame


def make_dynamics_patched(predict_sequence=False):
    return DynamicsModel(
        vocab_size=VOCAB, obs_h=OBS_H, obs_w=OBS_W, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, context_length=CONTEXT,
        latent_dim=LATENT_DIM, predict_sequence=predict_sequence,
        horizon=HORIZON if predict_sequence else 1, patch_size=PATCH,
    )


@torch.no_grad()
def test_dynamics_patch_single_frame():
    dyn = make_dynamics_patched()
    logits = dyn(history(), action())
    assert logits.shape == (BATCH, S, VOCAB)


@torch.no_grad()
def test_dynamics_patch_sequence_teacher():
    dyn = make_dynamics_patched(predict_sequence=True)
    logits = dyn(history(), action(), horizon=HORIZON, teacher_frames=frame_sequence(HORIZON))
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)


@torch.no_grad()
def test_dynamics_patch_sequence_autoregressive():
    dyn = make_dynamics_patched(predict_sequence=True)
    logits = dyn(history(), action(), horizon=HORIZON)
    assert logits.shape == (BATCH, HORIZON, S, VOCAB)


@needs_cuda
def test_dynamics_patch_backward():
    dyn = make_dynamics_patched().cuda()
    logits = dyn(history().cuda(), action().cuda())
    target = torch.randint(0, VOCAB, (BATCH, S), device="cuda")
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()


# --------------------------------------------------------------------------- #
# --- EMAEncoder ------------------------------------------------------------ #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def test_ema_encoder_encode_shape():
    enc = make_encoder(JEPAEncoder)
    ema = EMAEncoder(enc)
    out = ema.encode(single_frame())
    assert out.shape == (BATCH, LATENT_DIM)


@torch.no_grad()
def test_ema_encoder_update_changes_weights():
    enc = make_encoder(JEPAEncoder)
    ema = EMAEncoder(enc, decay=0.0)  # decay=0 → EMA params fully replaced by online
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(torch.ones_like(p))
    before = next(ema.encoder.parameters()).clone()
    ema.update(enc)
    after = next(ema.encoder.parameters())
    assert not torch.allclose(before, after)
