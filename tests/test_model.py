import pytest

torch = pytest.importorskip("torch")

from tuneproof.experiment import ExperimentConfig, fit  # noqa: E402
from tuneproof.model import (  # noqa: E402
    EOS,
    PAD,
    ModelConfig,
    TinyLM,
    encode_batch,
    parameter_hash,
    response_loss,
)

pytestmark = pytest.mark.ml


@pytest.fixture(autouse=True)
def setup_torch():
    torch.manual_seed(11)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    return TinyLM(ModelConfig(width=16, heads=2, layers=1, context=48, rank=2))


def test_response_mask_and_eos():
    x, y = encode_batch([("hi", "OK"), ("longer", "X")], 48)
    assert y[0].tolist()[:6] == [-100, -100, -100, ord("O"), ord("K"), EOS]
    assert (y[0, 6:] == -100).all()
    assert (x[0, 6:] == PAD).all()
    assert int((y != -100).sum()) == 5


def test_utf8_bytes_not_char_count():
    _, y = encode_batch([("Q", "ә")], 48)
    assert int((y != -100).sum()) == len("ә".encode()) + 1


def test_causal_attention_cannot_see_future(model):
    ids = torch.tensor([[1, 2, 3, 4]])
    changed = torch.tensor([[1, 2, 200, 201]])
    assert torch.equal(model(ids)[:, :2], model(changed)[:, :2])
    assert not torch.equal(model(ids)[:, 2:], model(changed)[:, 2:])


def test_padding_does_not_change_real_token_logits(model):
    x, _ = encode_batch([("hi", "OK")], 48)
    padded = torch.cat([x, torch.full((1, 5), PAD)], dim=1)
    assert torch.allclose(model(x), model(padded)[:, : x.shape[1]], atol=1e-6)


def test_initial_lora_is_identity(model):
    x, _ = encode_batch([("Q", "A")], 48)
    model.mode(False)
    original = model(x).detach()
    model.mode(True)
    assert torch.equal(original, model(x))


def test_lora_updates_only_adapter(model):
    model.mode(True)
    frozen = parameter_hash(model, adapter=False)
    adapter = parameter_hash(model, adapter=True)
    loss_before = float(response_loss(model, [("Q", "A")]).detach())
    fit(model, [("Q", "A")], steps=15, batch_size=2, lr=0.01, seed=5, phase="test")
    assert parameter_hash(model, adapter=False) == frozen
    assert parameter_hash(model, adapter=True) != adapter
    assert float(response_loss(model, [("Q", "A")]).detach()) < loss_before


def test_zero_lr_preserves_every_parameter(model):
    model.mode(True)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    fit(model, [("Q", "A")], steps=3, batch_size=2, lr=0, seed=5, phase="control")
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("pairs,context", [([], 48), ([("Q", "")], 48), ([("x" * 50, "A")], 48)])
def test_invalid_batches(pairs, context):
    with pytest.raises(ValueError):
        encode_batch(pairs, context)


def test_context_limits(model):
    with pytest.raises(ValueError, match="context"):
        model(torch.ones((1, 49), dtype=torch.long))
    with pytest.raises(ValueError, match="context"):
        model.generate("Q" * 48)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 17},
        {"heads": 0},
        {"layers": 0},
        {"context": 2},
        {"rank": 100},
        {"alpha": float("nan")},
    ],
)
def test_model_config_rejects_bad_inputs(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"adapter_steps": 0},
        {"pretrain_steps": 5001},
        {"learning_rate": float("nan")},
        {"learning_rate": -1},
        {"threads": 0},
        {"batch_size": 129},
    ],
)
def test_experiment_config_rejects_bad_inputs(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)
