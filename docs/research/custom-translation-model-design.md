# Custom Translation Model Design Document

**Date:** 2026-04-04
**Status:** Research / Planning
**Goal:** Design a small, locally-run EN->PL translation model optimized for Apple Silicon, targeting 2-3 GB memory budget for movie subtitle translation.

---

## 1. Apple Silicon AI Hardware Capabilities

### M1 MacBook Air (current hardware)

| Compute Unit | FP32 | FP16 | BF16 | INT8 | INT4 | FP8 | Peak Throughput |
|---|---|---|---|---|---|---|---|
| GPU (7-core) | Yes | Yes (2x FP32) | **No** | In shaders | Software dequant | No | 2.3 / 4.6 TFLOPS (FP32/FP16) |
| Neural Engine (16-core) | No | Yes | **No** | Weight-only* | Weight-only* | No | 11 TOPS (INT8), ~5.5 TFLOPS (FP16) |
| AMX (CPU matrix) | Yes | Yes | **No** | Yes | No | No | ~1.5 TFLOPS FP32 |

\* ANE dequantizes INT8/INT4 weights to FP16 before computation -- no actual INT8 compute throughput gain.

- **Memory bandwidth: 68.25 GB/s** -- the primary bottleneck for LLM inference.
- **No BF16 anywhere** (ARMv8.5-A). MLX defaults to BF16, incurring software emulation penalty on M1.
- Token generation is memory-bandwidth-bound: `tokens/sec ~ bandwidth / model_size_in_GB`.

### M5 (Oct 2025 base / Mar 2026 Pro+Max)

| Compute Unit | FP32 | FP16 | BF16 | INT8 | INT4 | FP8 | Peak Throughput |
|---|---|---|---|---|---|---|---|
| GPU ALUs (10-core) | Yes | Yes (2x) | No | In shaders | Software | No | 4.15 / 8.3 TFLOPS |
| **GPU Neural Accelerators (NEW)** | FP16->FP32 | Yes | Unclear | **Yes (2x FP16)** | No | **No** | ~70 TFLOPS FP16 (Max variant) |
| Neural Engine (16-core) | No | Yes | No | Weight-only | Weight-only | No | ~38-50 TOPS |
| AMX/SME (CPU) | Yes | Yes | **Yes (via SME)** | Yes | No | No | ~2+ TFLOPS |

**Key M5 innovations:**
- **GPU Neural Accelerators** in every GPU core -- Apple's first dedicated tensor/matrix units (analogous to NVIDIA Tensor Cores). Separate datapath from ALUs, purpose-built for matrix multiplication.
- INT8->INT32 matmul runs at **2x FP16 throughput** in Neural Accelerators -- genuine hardware INT8 acceleration.
- Memory bandwidth: **153.6 GB/s** (base), 614 GB/s (Max) -- 2.3x over M1.
- BF16 available in CPU AMX/SME path (first Apple chip with ARM SME was M4).
- **NOT added:** FP8, native INT4, structured sparsity, dedicated BF16 in Neural Accelerators.

### M5 vs M4 vs M1 Real Benchmarks (MLX, Apple's paper)

| Workload Type | M5 vs M4 | M5 vs M1 (estimated) |
|---|---|---|
| Prompt processing (compute-bound) | **3.5-4x faster** | ~8-10x faster |
| Token generation (bandwidth-bound) | **1.2-1.3x faster** | ~2-2.5x faster |
| Image generation (FLUX 12B 4-bit) | **3.8x faster** | N/A |

### Memory Bandwidth Comparison

| Chip | Bandwidth | Theoretical tok/s (1.3B INT8, ~1.3GB) | Theoretical tok/s (7B Q4_K_M, ~4GB) |
|---|---|---|---|
| M1 | 68 GB/s | ~52 | ~17 |
| M4 | 120 GB/s | ~92 | ~30 |
| M4 Pro | 273 GB/s | ~210 | ~68 |
| M4 Max | 546 GB/s | ~420 | ~137 |
| M5 | 154 GB/s | ~118 | ~39 |
| M5 Max | 614 GB/s | ~472 | ~154 |

Real-world achieves ~50-70% of theoretical due to KV cache, attention overhead, framework costs.

---

## 2. Quantization Performance for LLM Inference

### Decode Speed (tokens/sec) -- 7B model, batch=1

| Quantization | Model Size | M1 (68 GB/s) | M4 (120 GB/s) | M5 (154 GB/s) | M4 Max (546 GB/s) |
|---|---|---|---|---|---|
| Q4_K_M | ~4 GB | 12-15 | 20-25 | ~26-32 | 75-90 |
| Q5_K_M | ~5 GB | 10-12 | 17-21 | ~22-27 | 60-75 |
| Q8_0 | ~6.7 GB | 8-10 | 14-17 | ~18-22 | 50-60 |
| F16 | ~13 GB | 5-6 | 8-10 | ~10-13 | 30-38 |

### Smaller Models (1-3B, Q4_K_M)

| Model Size | Weight Size | M1 | M4 | M4 Pro |
|---|---|---|---|---|
| 1B | ~0.6 GB | ~60-70 | ~100-120 | ~200+ |
| 3B | ~1.8 GB | ~30-40 | ~50-65 | ~100-120 |

### Quantization Quality Impact for Translation

| Format | Bits/weight | Quality vs FP16 | Translation Suitability |
|---|---|---|---|
| FP16 | 16.0 | Baseline | Excellent |
| INT8 | 8.0 | <1% drop | **Excellent -- best for translation** |
| Q4_K_M | 4.8 (mixed) | Minor degradation | Acceptable for general; risky for translation |
| INT4 | 4.0 | 4-8% degradation | **Risky** -- translation quality suffers more than chat |

**Critical finding:** Encoder-decoder models tolerate quantization much better than decoder-only. Research shows "W4A4 quantization introduces no to negligible accuracy degradation for encoder-decoder models, but causes significant accuracy drop for decoder-only."

---

## 3. Architecture Decision: Encoder-Decoder

At the 600M-1.3B scale, encoder-decoder wins for translation:

| Factor | Encoder-Decoder | Decoder-Only |
|---|---|---|
| Translation quality (same params) | **+5-9 BLEU** | Baseline |
| First-token latency | **47% lower** | Baseline |
| Throughput | **4.7x higher** | Baseline |
| Quantization resilience | **Near-zero loss at INT4** | Significant drop |
| Example models | NLLB-200, BiDi, Marian | TranslateGemma, HY-MT |

Exception: Highly specialized decoder-only models trained exclusively for translation (like DIETA) can match encoder-decoder quality, but require training from scratch.

---

## 4. Model Sizing for 2-3 GB Budget

| Precision | Params in 2 GB | Params in 3 GB |
|---|---|---|
| FP32 | ~500M | ~750M |
| FP16 | ~1B | ~1.5B |
| INT8 | ~2B | ~3B |
| INT4 | ~4B | ~6B |

### Candidate Models

| Model | Params | Architecture | FP16 | INT8 | INT4 | EN->PL Quality | License |
|---|---|---|---|---|---|---|---|
| Allegro BiDi-eng-pol | 209M | Enc-Dec (Marian) | 400MB | 200MB | 100MB | COMET-DA 86.2 | Apache 2.0 |
| NLLB-200 distilled 600M | 600M | Enc-Dec | 1.2GB | 600MB | 300MB | COMET-DA ~87 | CC-BY-NC-4.0 |
| **NLLB-200 distilled 1.3B** | **1.3B** | **Enc-Dec** | **2.6GB** | **1.3GB** | **650MB** | **COMET-DA ~88** | CC-BY-NC-4.0 |
| NLLB-200 3.3B | 3.3B | Enc-Dec | 6.6GB | 3.3GB | 1.7GB | COMET-DA ~89 | CC-BY-NC-4.0 |
| TranslateGemma 4B | 4B | Decoder-only | 8GB | 4GB | 3GB | COMET ~80-84 | Gemma |
| HY-MT1.5-1.8B | 1.8B | Decoder-only | 3.6GB | 1.8GB | 900MB | Good but **no Polish** | Apache 2.0 |

**Recommended:** NLLB-200 distilled 1.3B at INT8 (~1.3 GB deployed). Best quality-per-byte for EN->PL.

---

## 5. Inference Framework Comparison

| Framework | Enc-Dec Support | Apple Silicon Opt | Quantization | Speed vs PyTorch | Best For |
|---|---|---|---|---|---|
| **CTranslate2** | Excellent | ARM64 Ruy + Accelerate | INT8, INT16 | **2-8x faster** | Translation (enc-dec) |
| **MLX** | Limited | Native Metal | 4-bit, 8-bit | ~2-4x faster | Decoder-only LLMs |
| **llama.cpp** | Developing | Metal backend | GGUF Q2-Q8, F16 | ~2-4x faster | Decoder-only LLMs |
| **CoreML** | Works (conversion needed) | Neural Engine + GPU | FP16, W4A16 | HW accelerated | Vision, Whisper, small models |

CTranslate2 is the clear winner for encoder-decoder translation models. Pre-converted NLLB: `OpenNMT/nllb-200-distilled-1.3B-ct2-int8`.

---

## 6. M5-Specific Optimizations

If targeting M5 exclusively, INT8 becomes especially attractive:
- GPU Neural Accelerators have **native INT8->INT32 matmul at 2x FP16 throughput**.
- On M1, INT8 runs via CPU (CTranslate2 Ruy backend) -- still fast, but no GPU tensor acceleration.
- On M5, INT8 inference could leverage Metal 4 Tensor APIs for GPU-accelerated INT8 matmul -- **genuine hardware speedup, not just memory savings**.

### M1 vs M5 Decision

For 1.3B INT8 encoder-decoder model:
- **M1 works fine.** 1.3 GB fits in 8 GB RAM. CTranslate2 CPU path: ~50-100 sentences/sec batched.
- **M5 is 2-3x faster** for this workload. ~150-300 sentences/sec batched.
- Dropping M1 only justified if using 4B+ decoder-only models or needing Neural Accelerator INT8 path.

---

## 7. Recommended Design

### Target Model

**NLLB-200 distilled 1.3B**, fine-tuned on OpenSubtitles EN-PL (59.7M parallel sentences), quantized to INT8 via CTranslate2.

| Property | Value |
|---|---|
| Architecture | Encoder-decoder transformer |
| Parameters | 1.3B |
| Deployed size | ~1.3 GB (INT8) |
| Runtime memory | ~1.5-2 GB |
| Quality target | COMET-DA ~90+ (after fine-tuning) |
| Inference framework | CTranslate2 |
| Quantization | INT8 (best quality/size for translation) |
| Hardware target | M1+ (M5 preferred for INT8 tensor acceleration) |

### Fine-Tuning Data

- **OpenSubtitles EN-PL**: 59.7M parallel sentence pairs from movie/TV subtitles (via Helsinki-NLP/OPUS).
- Domain-specific: directly relevant to movie subtitle translation.
- NLLB was NOT trained specifically on OpenSubtitles, making it a good fine-tuning candidate.
- Expected improvement: +4-12 BLEU from domain adaptation.

### Implementation Phases

1. **Phase 1 -- Quick win:** Convert existing BiDi 209M to CTranslate2 INT8 (200MB, 2-4x speedup, same quality).
2. **Phase 2 -- Quality upgrade:** Switch to NLLB 1.3B via CTranslate2 INT8 (1.3GB, COMET ~88).
3. **Phase 3 -- Domain adaptation:** Fine-tune NLLB 1.3B on OpenSubtitles EN-PL, re-quantize to INT8. Target COMET ~90+.
4. **Phase 4 (optional) -- M5 optimization:** Port to Metal 4 Tensor APIs for GPU-accelerated INT8 matmul if CTranslate2 adds M5 Neural Accelerator support, or build custom MLX encoder-decoder path.

---

## 8. Hardware Data Types Summary

### What matters for this model

| Data Type | M1 Hardware Support | M5 Hardware Support | Relevance |
|---|---|---|---|
| **INT8** | ANE (dequant only), AMX | **GPU Neural Acc (native 2x)**, ANE, AMX/SME | **Primary target** -- best quality/perf for translation |
| FP16 | GPU (native), ANE, AMX | GPU, Neural Acc, ANE, AMX | Fallback / baseline |
| BF16 | **Not supported** | AMX/SME only, GPU unclear | Avoid on M1; marginal benefit over FP16 |
| INT4 | Software dequant only | Software dequant only | Not recommended for translation (quality loss) |
| FP8 | Not supported | **Not supported** | Not available on any Apple Silicon |
| FP32 | GPU, AMX | GPU, AMX | Training only; too large for deployment |

### What Apple Silicon does NOT support (as of M5)

- FP8 (no Apple chip has this; NVIDIA-only)
- Native INT4 compute (always software dequant to FP16/INT8)
- Structured sparsity (NVIDIA 2:4 since Ampere; Apple has nothing equivalent)
- BF16 in GPU ALUs or Neural Accelerators (only in CPU AMX/SME since M4)
