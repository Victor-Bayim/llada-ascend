# Maintenance changes

The maintained entry points live in `analysis/`; immutable snapshots live in
`archive/original/`. `batched-topk.patch` records the known algorithm fix, already
applied to the maintained sampler. Do not apply it to the preserved archive.

When different batch rows require different k, taking a prefix of an unsorted max-k
result can select the wrong subset. The maintained vectorized path uses sorted=True.
