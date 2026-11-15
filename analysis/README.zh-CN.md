[English](README.md) | **简体中文**

# 可运行的扩散推理路径

- `llada_torch_npu/ceval_diffusion.py`：共享维护版采样器与 CEval 评测器。
- `llada_torch_npu/bench_diffusion.py`：同步计时与 JSON 元数据。
- `llada_torch_npu/ceval_diffusion_subset.py`：小规模质量烟测。
- `llada_torch_npu/llada_diffusion_cli.py`：交互式 prompt 入口。
- `llada_torch_npu/runtime.py`：显式设备检查，不会静默从 NPU 回退至 CPU。
- `final_report/`：维护版形状矩阵与烟测 shell 包装脚本。

从仓库根目录使用 `python -m analysis.llada_torch_npu.<entry>`。
实际推理需要模型权重、HF 推理依赖和兼容的加速器环境；CPU 张量测试不需要模型或 HF 依赖。参见根目录 README。
包括实验性 xLLM 服务在内的原始源码保留在 `archive/original/`。
