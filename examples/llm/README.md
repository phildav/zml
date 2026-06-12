# LLM

`//examples/llm` runs interactive or one-shot text generation from a model repository from HuggingFace.

We support the following models, automatically detected from the `model_type` in the `config.json`:

- Llama 3.1
- Qwen 3.5
- LFM 2.5

## Run

To load a model from HuggingFace directly:

```bash
# CPU
bazel run //examples/llm -- --model=hf://meta-llama/Llama-3.1-8B-Instruct
# CUDA
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=hf://meta-llama/Llama-3.1-8B-Instruct
# ROCm
bazel run //examples/llm --@zml//platforms:rocm=true -- --model=hf://meta-llama/Llama-3.1-8B-Instruct
```

From a local directory:

```bash
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=/var/models/meta-llama/Llama-3.1-8B-Instruct/
```

For a single non-interactive prompt:

```bash
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=hf://meta-llama/Llama-3.1-8B-Instruct --prompt="Write a haiku about Zig"
```

## Multimodal

Run multimodal models like Qwen3-VL with `--image`.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG" width="420" alt="Candy with animal logo">
</div>

```bash
bazel run //examples/llm --@zml//platforms:cuda=true -- \
  --model=hf://Qwen/Qwen3-VL-2B-Instruct \
  --image=candy.jpg \
  --max-patches=3000 \
  --prompt="What animal is on the candy?"
```

```
Based on a close examination of the candies in the image, the animal depicted on the candy is a **turtle**.
```

## Options

- `--model=<path>`: Required. Model repository to load. This can be a local path or a huggingface/S3 URI such as `hf://...` or `s3://...`.
- `--prompt=<string>`: Optional. Runs a single prompt instead of opening the interactive chat loop.
- `--image=<path>`: Optional. Path to the image to use for generation (default: none).

- `--seqlen=<number>`: Optional. Maximum sequence length. Defaults to `2048`.
- `--max-patches=<number>`: Optional. Maximum number of patches to use for the visual encoder (default: `1024`).
- `--backend=<vanilla|cuda_fa2|cuda_fa3>`: Optional. Attention backend. If omitted, the program auto-selects one for the current platform.
