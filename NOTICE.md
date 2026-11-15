# Attribution

Original source comments and copyright notices are preserved. The diffusion sampler follows the LLaDA model's published sampling approach; xLLM context files retain their upstream attribution. Victor's project work concerns platform adaptation and execution/memory handling, not authorship of the model or framework.

No new blanket license is assigned to third-party source in this evidence collection. Model weights, CEval data and proprietary accelerator runtimes are excluded. Consult the respective upstream projects for their licenses.

The CPU validation harness, data transcription and maintenance patch were prepared during the portfolio evidence review on 2026-11-01. They are not historical benchmark outputs.

On 2026-11-15 the source was restored into this public repository with a maintained
Python execution path, explicit device/synchronization checks and additional CPU
regressions. The unmodified competition snapshots remain under archive/original.
These restoration changes and CPU results do not claim new NPU performance or
quality measurements, or a validated xLLM C++ build. The historical upstream Git
base and complete accelerator software stack are unknown.
