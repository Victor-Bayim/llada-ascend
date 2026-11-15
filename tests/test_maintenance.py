"""CPU regressions of maintained functions and strict benchmark device handling."""
from types import SimpleNamespace
import torch
import pytest
from analysis.llada_torch_npu import ceval_diffusion as diffusion
from analysis.llada_torch_npu import runtime, bench_diffusion

torch.set_num_threads(1)


class ToyModel(torch.nn.Module):
    """Small deterministic projection/backbone, not a substitute for model quality."""
    def __init__(self, bias=True, vocab=37, hidden=13):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=bias)

    @property
    def device(self):
        return self.embedding.weight.device

    def model(self, input_ids, **kwargs):
        position = torch.arange(input_ids.shape[1], dtype=torch.float32)[None, :, None]
        return (self.embedding(input_ids) + position * .1,)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids)[0]))


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("chunk", [1, 8, 37, 64])
@pytest.mark.parametrize("mask_kind", ["none", "mixed", "empty", "all"])
@pytest.mark.parametrize("keep_block", [False, True])
@pytest.mark.parametrize("approx", [False, True])
def test_chunked_full_and_masked_head_agree(bias, chunk, mask_kind, keep_block, approx):
    torch.manual_seed(31)
    model = ToyModel(bias=bias)
    ids = torch.randint(0, 37, (2, 7))
    masks = {"none": None, "mixed": torch.rand(2, 7) > .5,
             "empty": torch.zeros(2, 7, dtype=torch.bool),
             "all": torch.ones(2, 7, dtype=torch.bool)}
    mask = masks[mask_kind]
    keep = torch.zeros(2, 7, dtype=torch.bool) if keep_block else None
    if keep is not None:
        keep[:, 2:5] = True
    actual, confidence = diffusion._compute_logits_stats_chunked(model, ids, mask, keep, 0., chunk, approx)
    expected, full_confidence = diffusion._compute_logits_stats_full(model, ids, mask, keep, 0., approx, False)
    assert torch.equal(actual, expected)
    torch.testing.assert_close(confidence, full_confidence, rtol=3e-5, atol=3e-6)
    chosen = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask & (keep if keep is not None else True)
    assert torch.equal(actual[~chosen], ids[~chosen])
    assert torch.all(confidence[~chosen] == 0)


@pytest.mark.parametrize("chunk,masked,block", [(0, False, False), (5, False, False),
                                                (0, True, False), (5, True, True)])
@pytest.mark.parametrize("vectorized,where_update", [(False, False), (True, True)])
def test_synthetic_end_to_end_generation(chunk, masked, block, vectorized, where_update):
    torch.manual_seed(17)
    model = ToyModel(vocab=17).eval()
    with torch.no_grad():
        model.lm_head.weight[12].zero_()
        model.lm_head.bias[12] = -100
    ids = torch.tensor([[1, 2], [3, 4]])
    common = dict(steps=4, gen_length=4, block_length=2, temperature=0., cfg_scale=0., mask_id=12)
    expected = diffusion.diffusion_generate(model, ids, **common)
    actual = diffusion.diffusion_generate(model, ids, **common, logits_chunk_size=chunk,
        masked_lm_head=masked, masked_block_only=block, vectorized_topk=vectorized,
        use_where_update=where_update)
    assert torch.equal(actual, expected)
    assert actual.shape == (2, 6)
    assert torch.equal(actual[:, :2], ids)
    assert not torch.any(actual[:, 2:] == 12)


def test_heterogeneous_topk_and_zero_rows():
    for seed in range(100):
        torch.manual_seed(seed)
        confidence = torch.randn(3, 1000)
        counts = torch.tensor([[100], [7], [0]])
        expected = diffusion._build_transfer_index(confidence, counts, 0, False, None)
        buffer = torch.ones_like(confidence, dtype=torch.bool)
        actual = diffusion._build_transfer_index(confidence, counts, 0, True, buffer)
        assert torch.equal(actual, expected)
        assert actual.data_ptr() == buffer.data_ptr()
        assert torch.equal(actual.sum(1), counts[:, 0])


def test_explicit_cpu_and_rejected_accelerator():
    assert runtime.resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError):
        runtime.resolve_device("cuda:0")


def test_missing_npu_extension_is_error(monkeypatch):
    def missing(_):
        raise ImportError("not installed")
    monkeypatch.setattr(runtime.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="No CPU fallback"):
        runtime.resolve_device("npu:0")


def test_unavailable_npu_is_error(monkeypatch):
    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: None)
    monkeypatch.setattr(runtime, "torch", SimpleNamespace(npu=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="No CPU fallback"):
        runtime.resolve_device("npu:0")


def test_synchronization_failures_are_not_hidden(monkeypatch):
    def fail():
        raise RuntimeError("NPU synchronize failed")
    monkeypatch.setattr(runtime, "torch", SimpleNamespace(npu=SimpleNamespace(synchronize=fail)))
    with pytest.raises(RuntimeError, match="synchronize failed"):
        runtime.synchronize_device(SimpleNamespace(type="npu"))


def test_benchmark_checks_device_before_loading_model(monkeypatch):
    def fail(_):
        raise RuntimeError("device unavailable before model loading")
    monkeypatch.setattr(bench_diffusion, "resolve_device", fail)
    with pytest.raises(RuntimeError, match="before model loading"):
        bench_diffusion.run_benchmark(SimpleNamespace(device="npu:0"))


def test_cpu_cannot_accept_npu_tuning_flags():
    with pytest.raises(ValueError):
        runtime.configure_npu(torch.device("cpu"), jit_compile=True)
