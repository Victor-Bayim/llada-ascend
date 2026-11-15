# Maintenance changes

The maintained entry points live in `analysis/`; immutable snapshots live in
`archive/original/`. `batched-topk.patch` records the known algorithm fix, already
applied to the maintained sampler. Do not apply it to the preserved archive.

When different batch rows require different k, taking a prefix of an unsorted max-k
result can select the wrong subset. The maintained vectorized path uses sorted=True.
The original report script enabled vectorized top-k only for batch=1, so this fix
must not be counted as a historical batch=16 timing improvement.

Additional restoration changes (2026-11-15):

- Model/dataset imports are lazy, so CPU tensor regressions run without weights or HF packages.
- NPU requests fail when the extension/device is unavailable. Device setup, tuning
  and synchronization errors propagate. Explicit `--device cpu` is a distinct functional run.
- Benchmarks synchronize before/after timing, use perf_counter, retain individual
  timings and emit requested flags, actual device/dtype and Python package versions.
- Runners accept a seed and optional model revision; maintained runners no longer
  force os._exit. The CLI's --devices alias accepts a single device.
- Shape-matrix/quality shell scripts run from a configured environment instead of
  requiring the unpinned historical privileged Docker command. New timings default
  to five runs; original one-run settings remain documented and archived.

CPU checks do not validate Ascend kernels, full model quality, half-precision
equivalence or xLLM C++ serving. Historical NPU data is not remeasured here.
