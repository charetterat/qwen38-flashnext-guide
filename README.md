<div align="center">

# Before you download Qwen3.8-Flash-Next: RAM, quants & reported pitfalls

**A source-linked decision checklist for running Qwen3.8-Flash-Next on a consumer GPU.**
Compare what your RAM and disk actually need, avoid a hardware claim that does not hold up, and
find the reported failures that produce *no error message*. **No independent testing — none of
these numbers are ours.**

> ⛔ **This is not official.** It is not affiliated with Alibaba/Qwen, with Strata, or with
> any runtime listed here. The upstream project's own install guide is always the thing to
> follow to install; this is for deciding *whether* and *which*.
>
> ⛔ **The throughput numbers below are not ours.** They are one project's measurements on
> two specific machines. We have not re-run them.
>
> 📅 **Sources last checked: 2026-10-24.** Only the citations listed under
> `data/*/verification_sources` are checked. Each check is a reachability + keyword-presence
> test — it proves the page still contains the string that made it worth citing. It does **not**
> prove the conclusion drawn from it is still correct, and it does not cover citations used
> only in body text.

[![sources checked](https://img.shields.io/badge/19%20sources%20reachability--keyword%20checked-8a8a8a)](scripts/verify_sources.py)
[![license](https://img.shields.io/badge/data-CC%20BY--4.0-lightgrey)](LICENSE)
[![verify](https://img.shields.io/badge/verify-scripts%2Fverify_sources.py-blue)](scripts/verify_sources.py)

**Start here if you are deciding something:**

- **"Will it fit in what I have?"** → [Pick by RAM and VRAM](#pick-by-ram-and-vram) · [file sizes](#quantizations-with-real-file-sizes)
- **"Something is broken and there is no error"** → [Failure modes](#failure-modes)
- **"Is the reason it fits what everyone says?"** → [How a 12 GB card stores a 180B model](#how-a-12-gb-card-stores-a-180b-model)
- **"Which engine can even load it?"** → [Runtime evidence found](#runtime-evidence-foundnot-a-compatibility-test)

</div>

---

## Read this first: two parameter counts, and which one to plan with

When sizing a download, "125B" is the language model only. The released checkpoint contains
more:

| component | size | note |
|---|---|---|
| **Total (released weights)** | **180B** | exact: 179,999,981,459 parameters, per the safetensors metadata |
| Language model | 125B | the figure usually quoted; 6B activated per token |
| N-gram embedding | 51B | 20,000,000 n-gram entries, an unusually large table |
| MTP draft layer | 4B | 1 layer, used for speculative decoding |
| **BF16 on disk** | **360 GB** | 131 safetensors shards |

This is not pedantry. The N-gram table is the single most load-bearing component for
memory behaviour, and it is one of the most consequential failure modes reported by users — see
[the failure that only costs you speed](#the-one-that-only-costs-you-speed).

⛔ If you are sizing a download from "125B", check whether the figure you are using accounts
for the other 55B.

---

## Architecture at a glance

Everything below is from the official `config.json` and model card, not from commentary.

| property | value |
|---|---|
| Architecture class | `Qwen4ExpForConditionalGeneration` (`qwen4_exp`) |
| Layers | 48 |
| Layer pattern | `12 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Qwen Sparse Attention -> MoE))` |
| Experts per layer | 512 routed + 1 shared, **10 activated** |
| Expert intermediate size | 640 |
| Hidden size | 2560 |
| Attention heads | 24 Q / 2 KV, head dim 256 |
| Attention mix | 36 linear (Gated DeltaNet) + 12 full (Qwen Sparse Attention) |
| Gated residual | 4 branches, bottleneck rank 320 |
| Vocabulary | 248,320 (padded) |
| Context | **262,144 natively**, extensible to 1,000,000 |
| Modality | image-text-to-text |
| License | Qwen Community License 1.0 |

The layer count is what makes this model unusual: **36 of
48 layers do not use softmax attention at all.** That is the
sparse-attention design. What it changes is the *cost of context*, not the amount of weight
you have to store — see [how it fits](#how-a-12-gb-card-stores-a-180b-model).

---

## Runtime evidence found — not a compatibility test

| runtime | finding | evidence kind | level | detail |
|---|---|---|---|---|
| **llama.cpp** | found | merged_pr | L2 | PR #27742 merged 2026-08-27; implementation in `src/models/qwen4exp.cpp` |
| **ExLlamaV3** | found | source_identifier | L1 | 4 code hits: `exllamav3/architecture/qwen4_exp.py`, `qwen4_exp_mtp.py`, `architectures.py`, `qwen3_vl.py` |
| **vLLM** | found | source_identifier | L1 | 37 code hits, with separate AMD and NVIDIA backends under `vllm/models/qwen4_exp/` |
| **SGLang** | found | source_identifier | L1 | 25 code hits: `srt/models/qwen4_exp.py`, `qwen4_exp_mtp.py`, `srt/configs/qwen4_exp.py` |
| **Ollama** | found | release_artifact | L3 | one library tag, `qwen3.8-flash-next:125b-mlx` at 105 GB |
| **MLX-LM** | not found in search | no_finding | — | 0 code hits for `qwen4_exp` or `Qwen3.8` |
| **llamafile** | not found in search | no_finding | — | 0 code hits for `qwen4_exp`; 1 test script string match for `Qwen3.8` (excluded as non-implementation) |
| **LM Studio** | could not determine | no_finding | — | 0 code hits in the `lms` app repo, which does not vendor the model code |

**Evidence levels** — the ladder is *identifier found → PR merged → release shipped →
third party reported it running*. A code-search hit proves step 1 and nothing beyond it.

| level | means |
|---|---|
| **L1** | 1. identifier found in source (this is what a code hit means) |
| **L2** | 2. support PR merged upstream |
| **L3** | 3. a release artifact that actually contains it |
| **L4** | 4. a third party reported it running |

ⓘ How these were determined, and how to trust them: each row comes from a code-search hit
or a live registry entry, and **each was preceded by a control query against the same
repository** to prove the repo was indexed and the scanner could hit it. A zero from an
unproved scanner is not evidence. That control also caught two repository renames —
`turboderp/exllamav3` and `ggml-org/llamafile` both return 404; the live paths are
`turboderp-org/exllamav3` and `mozilla-ai/llamafile`.

Full evidence, per runtime: [`data/runtimes.json`](data/runtimes.json).

### What the matrix actually means for you

⚠️ Finding an identifier is not finding a working build. Read the level column before you
plan around a row.

- **Server or workstation, plenty of VRAM** → vLLM and SGLang each carry a dedicated
  `qwen4_exp` model package, and vLLM's has separate AMD and NVIDIA kernel paths. ExLlamaV3
  carries `qwen4_exp` architecture files and a n-gram conversion utility that the GGUF path
  does not need. ⓘ That is *identifier-in-source* evidence — see the caveat above.
- **One consumer GPU with 12–24 GB** → none of the runtimes surveyed here was found to fit the
  model without an offload strategy. ⛔ That is a limit of this survey, not a proof that none
  exists. The consumer path described below is Strata's.
- **Apple silicon** → Ollama serves exactly one tag, `qwen3.8-flash-next:125b-mlx`, at
  105 GB. There is no plain GGUF tag, and the official `qwen` library does not carry it.

---

## How a 12 GB card stores a 180B model

> ⚠️ Strata's README calls the 3-of-4-linear-attention layout a reason the model is tractable. That is not the memory mechanism. Linear/sparse attention lowers long-context compute and KV-cache cost; it does not reduce the weight bytes you have to store. The weight-capacity answer is: MoE sparsity (compute) + quantization (bytes per weight) + offload (which tier holds them).

| lever | what it is | what it buys |
|---|---|---|
| **MoE sparsity (compute, not memory)** | 48 layers x 512 experts = 24,576 experts, but only 10 routed + 1 shared are chosen per layer per token | about 6B parameters of compute per token instead of 180B. That is why the per-token work is small. It is NOT why the model fits, and small per-token work alone does not guarantee fast decoding once weights are being moved between storage tiers. |
| **quantization (bytes per weight)** | Q2_0 through IQ3_S store the experts at 2-3 bpw, so each per-quant shard 1 (the expert weights) lands at 37.6-54.8 GB instead of the BF16 expert-shard equivalent. The 360 GB BF16 figure is the entire released checkpoint including embedding and MTP, not the same component as shard 1, so it is not on the same comparison axis as the 37.6-54.8 GB numbers; we do not state a compression ratio from these. | this is what shrinks the resident set; it does not change the embedding / MTP components |
| **tiered offload (where it lives)** | Per Strata's description: a small hot set of expert weights on the GPU, the rest in system RAM, and the 28.8 GB n-gram lookup table mmap-ed on the SSD. Which slice sits where depends on the build, the driver, the runtime flags and what else is open; the 12 GB figure is the GPU's share at one configuration, not the model's total resident set. | this is what lets a 12 GB card be enough at one configuration - the card only ever holds a slice |
| **linear + sparse attention (context, not capacity)** | 36 of 48 layers use Gated DeltaNet (linear attention with recurrent state) rather than softmax attention, altering context state memory scaling; the remaining 12 layers use Qwen Sparse Attention with a 2048-token index budget, reducing per-token attention compute and memory bandwidth. | Reduces per-token attention computation and recurrent state growth in DeltaNet layers during long-context processing. Note that sparse attention does not necessarily reduce persistent KV storage without an explicit cache eviction policy in the runtime. Crucially, neither mechanism reduces the model's weight byte count on disk or in RAM. |

**Hardware floor** (from Strata's own requirements — upstream's claims, not ours):

| component | minimum | practical |
|---|---|---|
| GPU | NVIDIA GeForce RTX 20/30/40/50 series, or AMD Radeon RX 7900 XT/XTX, 7800 XT, 7700 XT, 9060 XT, 9070/9070 XT, Radeon AI PRO R9700, or RX 6800/6900 series. Needs 12 GB VRAM or more. | **12 GB VRAM** is the stated floor |
| RAM | 32 GB or more | Upstream recommends 64 GB for typical configurations. This guide does not measure peak host process memory or guarantee OS headroom, load success, or paging-free operation. |
| Disk | Strata states about 80 GB free, but that figure does not cover the largest size - budget per quant from the table instead, and treat that as the sum of the listed files, not a peak-space guarantee for the whole install. Use an SSD: the first start is much faster and the SSD-offload path depends on it. | SSD; the first start is much faster and the SSD-offload path depends on it |
| OS | Windows 10/11 or Linux with a current NVIDIA or AMD driver | two or three cards can share the model |

ⓘ Strata describes 35-55 GB into RAM at recommended settings (i.e. shard 1 plus a working set, not the full BF16 weights).
1-3 minutes while the model loads, and longer the first time. If it is still unresponsive after 10 minutes, that is not normal: close other programs, restart, or pick a smaller size.

### Pick by RAM and VRAM

Answer these in order, then read the matching row: (1) Will you write in Chinese or other CJK text? If yes, exclude the Coder entirely. (2) Do you need image input? If yes, add the ~1 GB vision encoder file and check your OS/GPU combination in the failure-modes table. (3) Will you wire this into a coding agent that uses tool calls? Read the two Strata issues on tool-call loss first. (4) Match by RAM and VRAM. (5) Match by disk: per-quant budgets are in the table below. The `take` column is Strata's recommendation, not a result we measured; treat the sizes as documented and the `why` as arithmetic on shard sizes rather than tested fit.

| RAM | VRAM | take | upstream says | caveat | task caveats |
|---|---|---|---|---|---|
| 32 GB | 12-16 GB | **Coder** | upstream recommends this size | the only published variant whose shard-1 file size (29.6 GB) is under 32 GB; real RAM/VRAM paging depends on runtime and OS headroom | Do NOT pick Coder if you work in Chinese or other CJK text — answers come back wrong or looping. Switching to full-expert quants on 32 GB RAM is a language-quality choice, not an assurance of viable performance. |
| 32 GB | 24 GB | **Q2_0 or IQ2_XS** | upstream documents it as a low-RAM exception (slower) | slower - the GPU holds more, the rest streams from the model files | Retains all experts (avoiding Coder's CJK degradation, though independent GPU issues like sm120 CUDA 13.2 still apply). If you need image input, also add the ~1 GB vision encoder file. |
| 48 GB | >= 12 GB | **IQ2_XS, or Q2_0 for speed** | upstream recommends this size | shard-1 file size is 39.2 GB; larger quants (47.0 GB and 54.8 GB) approach or exceed 48 GB | Retains all experts. Image input supported on NVIDIA on both Windows and Linux; on AMD it is CPU-only on Linux and not yet available on Windows (Strata #304). |
| 64 GB | >= 12 GB | **IQ2_XS** | upstream recommends this size | shard-1 file sizes range from 37.6 to 54.8 GB | Retains all experts. Tool calls in agent harnesses can be lost inside an unclosed <think> block (Strata #804) and silently dropped after an empty assistant turn (Strata #843). |
| 96 GB or more | >= 12 GB | **IQ3_S, or Unsloth UD-IQ4_XS** | upstream recommends this size | UD-IQ4_XS is 94 GB and reads from SSD during inference if RAM is constrained | Retains all experts. Best size for vision input — both NVIDIA and AMD paths are more responsive when host RAM is unconstrained. |

⛔ Each row in this table is Strata's recommendation, not a result we measured. Treat the
sizes as what upstream documents, and the "why" as arithmetic on shard sizes rather than a
tested fit. Read the "task caveats" column *before* matching by RAM.

---


## Quantizations, with real file sizes

These are the actual byte counts of the published files, not estimates. The
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) set is the
one the installer uses.

| quant | shard 1 file size — not measured memory usage | download | disk budget | speed |
|---|---|---|---|---|
| `Q2_0` | **37.6 GB** | 66.4 GB | 73 GB* | highest in this table (one project, two machines; see Measured throughput) |
| `IQ2_XS` ⭐ | **39.2 GB** | 68.0 GB | 75 GB* | recommended by Strata |
| `IQ3_XXS` | **47.0 GB** | 75.8 GB | 83 GB* | — |
| `IQ3_S` | **54.8 GB** | 83.6 GB | 90 GB* | — |

*disk budget = download plus the ~6 GB MTP draft layer, plus ~1 GB more if you enable the
vision encoder. Quality ordering is deliberately omitted: the `good / better / great / best`
labels in the upstream docs are the author's wording, and no independent benchmark for this
model on consumer hardware was identified in this survey.

The gap between the two size columns is the 28.8 GB second shard, which is identical across
all four variants. Upstream identifies it as the per-layer n-gram embedding table
(`per_layer_token_embd`, 51.2B parameters) held at a fixed 4.5 bpw — it is a lookup, not a
matmul weight, so it does not get quantized down with the rest. The publisher recommends
`-lm mmap --lazy-mode on` to keep shard 2 memory-mapped on SSD. That is why disk space and
RAM are separate constraints here.

### Other published variants

| publisher | what it is | status |
|---|---|---|
| Unsloth | Unsloth's ~4-bit line (1.37M downloads as of 2026-10-06, the second-highest HF download count among republishers of this model, after ISTA-DASLab at 2.24M) — 49.8-54.4 GB for the K-quants; 102.4 GB BF16 | UD-IQ4_XS is in the installer menu; UD-Q4_K_XL and UD-Q6_K_XL are experimental and not in the menu |
| UkisAI | fine-tune that thinks for a much shorter time before answering. Strata's docs say you get the answer sooner at about the same quality - their assessment, not a benchmark we ran or found —  | installer menu via --family swift |

### The Coder variant — and the trap in it

The [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) keeps
only **256 of each layer's 512, chosen on code (agentic + image) data**.
Upstream documents this variant for certain 32 GB RAM configurations based on its 29.6 GB shard-1 file size; peak process memory is not measured by this project, so zero-paging cannot be guaranteed. Its authors report
91% of the full model's SWE-bench Verified, 99% of LiveCodeBench (publisher's own numbers; not independently reproduced here).

⛔ **It is weaker outside code, and specifically on Chinese and other CJK text**, where
answers came out wrong or looping where English was fine
([Strata #438](https://github.com/Niko1221/Strata/issues/438)). If you work in Chinese, take
`Q2_0`, `IQ2_XS` or `IQ3_S` — the ones that keep every expert.

---

## Measured throughput

> These are Strata's published numbers on two specific machines, not independent verification, and they do not transfer to other cards. Neither table represents a controlled comparison: System A conflates quant size with engine version (0.1.36 vs 0.1.26), and System B's engine versions were not reported by upstream. Rows list full-expert quants in ascending shard-1 size, with the Coder variant (256 experts kept) appended as a specialized sub-variant rather than a speed ranking.

**System A** — NVIDIA RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM, engine 0.1.36 (Q2_0) / 0.1.26 (others), 4K answers, 32K prompts

> ⛔ **Not a controlled quantization comparison.** Q2_0 was measured on engine 0.1.36; the
> remaining rows were measured on engine 0.1.26. The spread across rows conflates the
> quantization effect with the engine-version effect. Full-expert quants are ordered by
> shard-1 file size; Coder is appended as a specialized sub-variant.

| quant | engine | decode (short chat) | decode (128K ctx) | prefill (32K prompt) |
|---|---|---:|---:|---:|
| `Q2_0` | 0.1.36 | 94 tok/s | 76 tok/s | 2,650 tok/s |
| `IQ2_XS` | 0.1.26 | 79 tok/s | 63 tok/s | 2,090 tok/s |
| `IQ3_XXS` | 0.1.26 | 62 tok/s | 49 tok/s | 1,750 tok/s |
| `IQ3_S` | 0.1.26 | 53 tok/s | 46 tok/s | 1,620 tok/s |
| `Coder (IQ1_M)` | 0.1.26 | 55 tok/s | 43 tok/s | 2,180 tok/s |

**System B** — AMD RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM, Linux

> ⛔ **Unverified engine versions.** The engine versions for System B were not reported in the
> upstream publication. Neither System B rows nor comparisons between System A and System B
> represent a controlled benchmark.

| quant | engine | decode (short chat) | decode (128K ctx) | prefill (32K prompt) |
|---|---|---:|---:|---:|
| `Q2_0` | unverified (not reported by upstream) | 60 tok/s | 48 tok/s | 1,160 tok/s |
| `IQ2_XS` | unverified (not reported by upstream) | 52 tok/s | 36 tok/s | 1,110 tok/s |
| `Coder (IQ1_M)` | unverified (not reported by upstream) | 44 tok/s | 33 tok/s | 1,420 tok/s |

ⓘ Writes answers = decode throughput. Reads your prompt = prefill throughput on a 32K-token prompt. A token is about 3/4 of a word in English; in Chinese/Japanese/Korean the ratio is closer to 1:1, so the same tok/s reads slower to a CJK reader.

⚠️ These are published shard-1 FILE SIZES, not measured peak process memory. Real usage also
depends on the runtime, the offload path, your context length and what else is open. Strata
describes its offload path as shard 1 resident in RAM with a working slice on the GPU and the
28.8 GB shard 2 mmap-ed on SSD; the proportions depend on the build, the driver and the
runtime flags, and a 12 GB card is enough because only a slice lives there.

### Projections — not measurements

Strata publishes these estimates and labels them as such. **Do not quote them as measured.**

| configuration | projected decode | why |
|---|---|---|
| RTX 3090 (24 GB), same original model | **roughly 100–140 tokens/s decode** | more VRAM means more of the model stays on the card |
| Unsloth UD-Q4_K_XL on a 64 GB PC with a 12 GB GPU | **7–8.5 tokens/s** | 77 GB of experts do not fit in RAM, so most are read from the SSD during inference |

### Community reports

| date | hardware | configuration | report |
|---|---|---|---|
| 2026-09-30 | RTX 5090, Core Ultra 9 285K, 64 GB RAM | Strata 0.1.29, original Flash-Next IQ2_XS, 131,072-token context | [read](https://github.com/Niko1221/Strata/blob/main/bench/results/2026-09-30-community-rtx-5090/README.md) |
| 2026-10-03 | 2x AMD Instinct MI50 16 GB (gfx906), Xeon E5-2666 v3, 32 GB RAM | the gfx906 build (#638) with #639 and #640, Coder IQ1_M, 131,072-token context, layer split across both cards | [read](https://github.com/Niko1221/Strata/blob/main/bench/results/2026-10-03-community-2x-mi50/README.md) |
| 2026-10-04 | 2x Intel Arc Pro B60 24 GB, Ryzen 5 5600, 64 GB RAM | the SYCL port at 6f32ec0 with two fixes, Coder IQ1_M and Flash-Next IQ2_XS, 8,192-token context | [read](https://github.com/Niko1221/Strata/blob/main/bench/results/2026-10-04-community-2x-arc-pro-b60/README.md) |

---

## Failure modes

Every item below is from a runtime that *does* support this model. These are not
"unsupported" problems; they are the specific things that break once you have it running.

ⓘ A note on the two worst ones: **they fail silently.** One produces correct-looking
answers at terrible speed. The other produces wrong answers with no error in the log.

### 🔴 The one that produces wrong answers

**Reported: correct on CPU, wrong on GPU — RTX 50-series (sm_120) with CUDA 13.2, on
`IQ1_S` / `IQ2_S` / `IQ3_S`.**

[RTX 50-series (sm_120): reported wrong answers with CUDA 13.2 on IQ1_S / IQ2_S / IQ3_S](https://github.com/Niko1221/Strata/issues/892) — [Strata #892](https://github.com/Niko1221/Strata/issues/892), open

> On Linux with CUDA 13.2 on an RTX 5070 Ti the engine loads and serves normally with no errors in the log, but every answer is wrong. 'What is 2+2?' returns '4+4=8.8.' Translating 'good morning' to German returns 'Good morning'. Most other prompts end after 1-2 tokens or loop.

The reporter's own isolation, in their words: the same build is correct with `-ngl 0` and wrong
with GPU offload, and `test-backend-ops` shows `MUL_MAT` 44 FAIL / `MUL_MAT_ID` 22 FAIL on
exactly those three quant types. ⓘ That localises it to the GPU path for those quants.

⛔ **We are not claiming we know the mechanism.** A compiler bug, a kernel implementation bug
and a driver interaction all remain consistent with what was reported. Nobody has ruled any of
them out. What is *not* speculative is the symptom: answers that look fine and are not.

**Fix:** No fix reported. The reporter states CUDA 13.0 works on the same card. ⚠️ *reporter-tested workaround, not a maintainer-confirmed fix.* Evidence level: 1 = reporter reproduction with an isolation matrix.

**Before you debug anything else:** if answers come back *wrong* rather than missing, and you
are on an RTX 50-series card, check your `nvcc` version before anything else. This fails
silently — the log stays clean.

### 🔴 The one that only costs you speed

**Reported: catastrophic prefill slowdown on newer prebuilt binaries, cause unconfirmed.**

[N-gram table reported as not resident on newer prebuilt binaries (cause unconfirmed)](https://github.com/ggml-org/llama.cpp/issues/28355) — [llama.cpp #28355](https://github.com/ggml-org/llama.cpp/issues/28355), closed

> On builds newer than 10665 the reporter observed the model loading and generating correct output while prefill was extremely slow. Their hypothesis, explicitly unconfirmed, is that the ~51B-parameter N-gram embedding table (20M bigram+trigram entries near layer 2) is not resident, so per-token lookups hit host memory and mmap page faults instead of being prefetched.

ⓘ The reporter did not report wrong answers. Nothing errored; prefill was the only thing affected. We are not claiming the outputs are correct in general - only that this report did not describe an output problem.

**Root cause status:** Reporter's hypothesis, explicitly unconfirmed. No maintainer statement found.

**Fix:** Closed. The reporter states the community prebuilt binary at build 10665 behaves correctly. No maintainer-confirmed root cause or fix version was found. Evidence level: 1 = reporter observation only.

If prefill is pathologically slow but output is correct, check build age before blaming the model or the quant.

### Memory growth on unified-memory systems

[Host RAM grows ~12 GiB/hour of normal chat until swap-thrash](https://github.com/ggml-org/llama.cpp/issues/28933) — [llama.cpp #28933](https://github.com/ggml-org/llama.cpp/issues/28933), open

On build b10909-mix-bea84f7 the llama-server process footprint grows continuously during ordinary multi-turn conversation, roughly +12 GiB per hour. On build b10830-mix-d5c17a0 it stays flat at ~2.1 GiB regardless of usage.

Reported process footprint, from the issue thread:

| build | what the reporter measured |
|---|---|
| build_a_stable | VmRSS 2.1 GiB, VmSwap 0, flat across days |
| build_b_fresh | VmRSS 19.2 GiB, VmSwap 9.6 GiB, RSS+Swap 28.8 GiB |
| build_b_after_1h | VmRSS 26.1 GiB, VmSwap 14.7 GiB, RSS+Swap 40.8 GiB |

**Fix:** No fix reported. Open at time of writing. The practical mitigation is the older build. Evidence level: 1 = reporter measurement, not independently reproduced.

⚠️ On the reporter's 128 GB unified-memory box this stopped being slow and became fatal - swap-thrash under load, twice, requiring a reboot. We are reporting that experience, not a general property of all such machines.

### Everything else

| what goes wrong | where | state | ev. | fix |
|---|---|---|---|---|
| **Coder variant answers wrong or loops on Chinese and other CJK text** | [Strata #438](https://github.com/Niko1221/Strata/issues/438) | reference in docs/MODELS.md | 2 | Use Q2_0, IQ2_XS or IQ3_S, which keep every expert, whenever you need CJK text. |
| **CPU vision path caps images at 300 tokens; llama.cpp's mtmd asks for >=1024** | [Strata #767](https://github.com/Niko1221/Strata/issues/767) | closed | 1 | Closed in Strata. Relevant if you are comparing the two paths on image input. |
| **Vision input on AMD: works on Linux through the CPU, unavailable on Windows** | [Strata #304](https://github.com/Niko1221/Strata/issues/304) | closed | 2 | No fix reported. Use text only, or an NVIDIA card for image input on Windows. |
| **Tool call written inside <think> without </think> is returned as reasoning_content** | [Strata #804](https://github.com/Niko1221/Strata/issues/804) | closed | 1 | Relevant if you are wiring this into a coding agent and tools appear to be ignored. |
| **Tool calls silently dropped in agent sessions after an empty assistant turn** | [Strata #843](https://github.com/Niko1221/Strata/issues/843) | closed | 1 | Closed with a fix in Strata. |
| **Frequent thinking loops** | [Strata #75](https://github.com/Niko1221/Strata/issues/75) | closed | 1 | — |
| **128K hard limit on 64 GB RAM is too conservative** | [Strata #406](https://github.com/Niko1221/Strata/issues/406) | closed | 2 | — |

⚠️ `closed` on GitHub means the thread ended, not that a fix shipped. Read the `fix` column,
not the `state` column.

---

## Start here

```bash
# Let an AI coding assistant set it up for you — this is the documented path
Set up Strata on this PC for me: https://github.com/Niko1221/Strata \
  - follow docs/AI_SETUP.md in that repository
```

Or by hand: download or clone Strata, then double-click `START-HERE.bat` on Windows or run
`./setup.sh` on Linux. Press Enter at each prompt for the recommended answer.

Then connect your tools with the OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1`.
Any API key and any model name work.

---

## Verify this repository

Every number above is compiled from project documentation and issue reports. To check the
cited pages have not gone stale:

```bash
python3 scripts/verify_sources.py
```

It fetches all 19
cited URLs and asserts each still contains the marker string that made it worth
citing. Four verdicts are kept deliberately distinct:

- `OK` — the page is live and still contains the token.
- `STALE` — the page is live but **the expected marker string was not found** in the cleaned text. This indicates the cited claim or phrasing may have changed and requires manual review.
- `UNREACHABLE` — the fetch failed after retries. We could not read the page; that is *not*
  evidence that the claim is still good, and not evidence that it went bad.
- `HTTP_ERROR` — a non-2xx HTTP status (such as 404 or 403). Treated as a gate failure requiring manual review.

⛔ The gate fails on an empty source list rather than passing vacuously, and it exits non-zero
on `STALE` so a drifted guide cannot pass CI.

---

## What this repository is not

- **Not a benchmark.** The throughput numbers are Strata's, on two specific machines.
  They do not transfer to your card.
- **Not a fork or mirror of any runtime.** It reads their docs and reports what they say.
- **Not exhaustive.** Quantization variants and issue reports appear daily. Check
  `data/*.json` for the last verified state.
- **Not independently benchmarked.** Every number on this page comes from a project that
  published it about itself. Nobody here has re-run a measurement.

### Corrections we have made to widely-repeated claims

This page is not neutral — it disagrees with some things in circulation. Each of these was
checked against the source named on the right:

| claim in circulation | what the source says | where |
|---|---|---|
| "125B model" | The released checkpoint is 180B (179,999,981,459 parameters). 125B is the language model only. Plan downloads against the larger figure. | [`model.safetensors.index.json`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/model.safetensors.index.json) |
| "3 of 4 layers use linear attention, that is why it fits on a small card" | The attention layout affects per-token compute and KV-cache cost, not the weight bytes you must store. What makes it fit is quantization + offload. | `config.json` vs. [Strata's own offload description](https://raw.githubusercontent.com/Niko1221/Strata/main/README.md) |
| "you need about 80 GB free" | That is Strata's own figure and it does not cover the largest size. Per-quant budgets are in the table above. | [ISTA-DASLab file listing](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main) |
| "the Coder is fine if you are not writing code" | It is specifically weak on Chinese and other CJK text, where answers came out wrong or looping. | [Strata #438](https://github.com/Niko1221/Strata/issues/438) |

⚠️ Two things this table is not. It is **not** an independent audit — we checked what the
sources say, not whether they are right. And "125B" is not necessarily *wrong*; it is a
different quantity from the one you need when sizing a download.

If you find another claim here that does not survive checking, that is a bug — open an issue
or read `data/*.json` to see exactly what was asserted and from where.

---

## Sources

| source | url |
|---|---|
| Official model card | `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/README.md` |
| config.json (authoritative architecture) | `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json` |
| safetensors index (authoritative BF16 size) | `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/model.safetensors.index.json` |
| ISTA-DASLab quant file listing | `https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main` |
| Strata README | `https://raw.githubusercontent.com/Niko1221/Strata/main/README.md` |
| Strata docs/MODELS.md | `https://raw.githubusercontent.com/Niko1221/Strata/main/docs/MODELS.md` |
| ISTA-DASLab Coder quant file listing | `https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/tree/main` |
| llama.cpp merged support PR | `https://github.com/ggml-org/llama.cpp/pull/27742` |
| ExLlamaV3 repo (correct org) | `https://github.com/turboderp-org/exllamav3` |
| vLLM qwen4_exp model package | `https://github.com/vllm-project/vllm/tree/main/vllm/models/qwen4_exp` |
| Ollama library page | `https://ollama.com/library/qwen3.8-flash-next` |
| llama.cpp issue 28355 (n-gram table) | `https://github.com/ggml-org/llama.cpp/issues/28355` |
| llama.cpp issue 28933 (memory growth) | `https://github.com/ggml-org/llama.cpp/issues/28933` |
| Strata issue 892 (wrong answers on GPU, CUDA 13.2) | `https://github.com/Niko1221/Strata/issues/892` |
| Strata issue 804 (tool call in think) | `https://github.com/Niko1221/Strata/issues/804` |
| Strata issue 304 (AMD vision) | `https://github.com/Niko1221/Strata/issues/304` |
| Strata issue 438 (Coder on CJK text) | `https://github.com/Niko1221/Strata/issues/438` |
| Strata issue 767 (CPU vision token cap) | `https://github.com/Niko1221/Strata/issues/767` |
| Strata issue 843 (tool calls dropped) | `https://github.com/Niko1221/Strata/issues/843` |

Licenses differ per component: the model is Qwen Community License 1.0, the quantizations
carry their publishers' terms, and Strata is MIT. Check each before you redistribute.

---

<div align="center">

**Data files** — [`data/model.json`](data/model.json) · [`data/runtimes.json`](data/runtimes.json) · [`data/failures.json`](data/failures.json)
**Scripts** — [`scripts/gen_readme.py`](scripts/gen_readme.py) · [`scripts/verify_sources.py`](scripts/verify_sources.py)

<sub>README.md is generated. Edit the JSON, not this file.</sub>

</div>
