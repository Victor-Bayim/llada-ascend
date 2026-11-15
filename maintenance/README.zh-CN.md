[English](README.md) | **简体中文**

# 维护改动

维护版入口位于 `analysis/`；不可变快照位于 `archive/original/`。
`batched-topk.patch` 记录已应用到维护版采样器的算法修复，不要再应用到保留的归档。

不同 batch 行需要不同 k 时，截取未排序 max-k 结果的前缀可能选错子集。
维护版向量化路径使用 sorted=True。
