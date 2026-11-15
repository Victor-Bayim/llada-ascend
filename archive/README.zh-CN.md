[English](README.md) | **简体中文**

# 原始源码上下文

`original/` 逐字节保留比赛源码材料中的 13 份文件，SHA-256 记录在 `../SOURCE-MANIFEST.json`。

原始 shell 脚本使用 xLLM `0.7.1-slim` 容器标签。

`original/xllm-src/` 包含四份上下文文件：LLaDA 注册/配置处理、Qwen3-MoE 集成、torch MoE 回退和单进程 worker 路径。
这**不是完整可构建的 xLLM 检出或经过验证的服务版本**。所引用的头文件与构建配置缺失，不能作为独立框架直接编译。

`original/analysis/llada_torch_npu/ceval_eval.py` 是实验性 xLLM 服务评测路径，作为归档保存，并非维护版扩散/延迟入口。
维护版 Python 入口直接使用 Transformers；在 xLLM 容器中运行不意味着结果来自 C++ 服务路径。

原始脚本包含历史 CPU 回退、未排序批量 top-k 和强制退出行为。维护版执行请使用 `analysis/`。
