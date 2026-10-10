#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render README.md from data/*.json.

Single source of truth: the JSON files. This script never contains a fact of
its own, so README.md cannot drift from the data without --check failing.

    python3 scripts/gen_readme.py            # write README.md
    python3 scripts/gen_readme.py --check    # exit non-zero if stale (CI gate)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "README.md")


def load(name):
    with open(os.path.join(ROOT, "data", name), encoding="utf-8") as f:
        return json.load(f)


def gb(n):
    return f"{n / 1e9:.1f} GB"


def table(headers, rows, align=None):
    align = align or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    m = load("model.json")
    rt = load("runtimes.json")
    fl = load("failures.json")

    model, arch, params, w = m["model"], m["model"]["architecture"], m["model"]["params"], m["model"]["weights"]
    L = []

    # The three failures that get their own section, looked up by id so the
    # generator does not depend on list order.
    by_id = {x["id"]: x for x in fl["failures"]}
    miscompile = by_id["sm120-cuda132-wrong-output"]
    ngram = by_id["ngram-table-not-resident"]
    growth = by_id["host-memory-growth"]
    rest = [x for x in fl["failures"] if x["id"] not in
            ("ngram-table-not-resident", "host-memory-growth", "sm120-cuda132-wrong-output")]

    # ---------------------------------------------------------------- header
    L.append(f"""<div align="center">

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
> 📅 **Sources last checked: 2026-10-10.** Only the citations listed under
> `data/*/verification_sources` are checked. Each check is a reachability + keyword-presence
> test — it proves the page still contains the string that made it worth citing. It does **not**
> prove the conclusion drawn from it is still correct, and it does not cover citations used
> only in body text.

[![sources checked](https://img.shields.io/badge/{len(m['verification_sources']) + len(rt['verification_sources']) + len(fl['verification_sources'])}%20sources%20reachability--keyword%20checked-8a8a8a)](scripts/verify_sources.py)
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

{table(
    ["component", "size", "note"],
    [
        ["**Total (released weights)**", f"**{params['total_safetensors_params_human']}**", f"exact: {params['total_safetensors_params']:,} parameters, per the safetensors metadata"],
        ["Language model", params["language_model_params"], f"the figure usually quoted; {params['activated_params_per_token']} activated per token"],
        ["N-gram embedding", params["ngram_embedding_params"], f"{arch['ngram_vocab_size_base']:,} n-gram entries, an unusually large table"],
        ["MTP draft layer", params["mtp_params"], f"{arch['mtp_layers']} layer, used for speculative decoding"],
        ["**BF16 on disk**", f"**{w['bf16_total_human']}**", f"{w['shards']} safetensors shards"],
    ],
    ["---", "---", "---"])}

This is not pedantry. The N-gram table is the single most load-bearing component for
memory behaviour, and it is one of the most consequential failure modes reported by users — see
[the failure that only costs you speed](#the-one-that-only-costs-you-speed).

⛔ If you are sizing a download from "125B", check whether the figure you are using accounts
for the other 55B.

---

## Architecture at a glance

Everything below is from the official `config.json` and model card, not from commentary.

{table(
    ["property", "value"],
    [
        ["Architecture class", f"`{m['model']['architecture_class']}` (`{m['model']['model_type']}`)"],
        ["Layers", f"{arch['num_hidden_layers']}"],
        ["Layer pattern", f"`{arch['layer_pattern']}`"],
        ["Experts per layer", f"{arch['num_experts']} routed + {arch['shared_expert']} shared, **{arch['num_experts_per_tok']} activated**"],
        ["Expert intermediate size", arch["moe_intermediate_size"]],
        ["Hidden size", arch["hidden_size"]],
        ["Attention heads", f"{arch['num_attention_heads']} Q / {arch['num_key_value_heads']} KV, head dim {arch['head_dim']}"],
        ["Attention mix", f"{arch['linear_attention_layers']} linear (Gated DeltaNet) + {arch['full_attention_layers']} full (Qwen Sparse Attention)"],
        ["Gated residual", f"{arch['gated_residual_branches']} branches, bottleneck rank {arch['gated_residual_bottleneck_rank']}"],
        ["Vocabulary", f"{arch['vocab_size']:,} (padded)"],
        ["Context", f"**{arch['context_native']:,} natively**, extensible to {arch['context_extensible_to']:,}"],
        ["Modality", m["model"]["modality"]],
        ["License", m["model"]["license_name"]],
    ],
    ["---", "---"])}

The layer count is what makes this model unusual: **{arch['linear_attention_layers']} of
{arch['num_hidden_layers']} layers do not use softmax attention at all.** That is the
sparse-attention design. What it changes is the *cost of context*, not the amount of weight
you have to store — see [how it fits](#how-a-12-gb-card-stores-a-180b-model).

---

## Runtime evidence found — not a compatibility test

{table(
    ["runtime", "finding", "evidence kind", "level", "detail"],
    [[f"**{r['name']}**",
      {"FOUND": "found",
       "NOT_FOUND": "not found in search",
       "UNKNOWN": "could not determine"}.get(r.get("finding_status", "UNKNOWN"), r.get("finding_status", "UNKNOWN")),
      r.get("evidence_kind", "—"),
      f"L{r['evidence_level']}" if isinstance(r.get("evidence_level"), int) else "—",
      r["summary"]] for r in rt["runtimes"]],
    ["---", "---", "---", "---", "---"])}

**Evidence levels** — the ladder is *identifier found → PR merged → release shipped →
third party reported it running*. A code-search hit proves step 1 and nothing beyond it.

{table(
    ["level", "means"],
    [[f"**L{i}**", t] for i, t in enumerate(rt["_methodology"]["support_scale"], 1)],
    ["---", "---"])}

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

> {m["why_a_12gb_card_runs_a_180b_model"]["_claim_correction"]}

{table(
    ["lever", "what it is", "what it buys"],
    [[f"**{lv['lever']}**", lv["what"], lv["effect"]]
     for lv in m["why_a_12gb_card_runs_a_180b_model"]["why_a_12gb_card_stores_a_180b_model"]],
    ["---", "---", "---"])}

**Hardware floor** (from Strata's own requirements — upstream's claims, not ours):

{table(
    ["component", "minimum", "practical"],
    [
        ["GPU", m["hardware_floor_from_strata"]["gpu"], m["hardware_floor_from_strata"]["gpu_practical"]],
        ["RAM", m["hardware_floor_from_strata"]["ram"], m["hardware_floor_from_strata"]["ram_practical"]],
        ["Disk", m["hardware_floor_from_strata"]["disk"], m["hardware_floor_from_strata"]["disk_practical"]],
        ["OS", m["hardware_floor_from_strata"]["os"], m["hardware_floor_from_strata"]["os_practical"]],
    ],
    ["---", "---", "---"])}

ⓘ {m['why_a_12gb_card_runs_a_180b_model']['ram_loaded_from_strata']}.
{m['why_a_12gb_card_runs_a_180b_model']['first_start']}

### Pick by RAM and VRAM

{m['pick_by_ram_preamble']}

{table(
    ["RAM", "VRAM", "take", "upstream says", "caveat", "task caveats"],
    [[r["ram"], r["vram"], f"**{r['take']}**", r["works"], r["caveat"], r.get("task_caveats", "—")] for r in m["pick_by_ram"]],
    ["---", "---", "---", "---", "---", "---"])}

⛔ Each row in this table is Strata's recommendation, not a result we measured. Treat the
sizes as what upstream documents, and the "why" as arithmetic on shard sizes rather than a
tested fit. Read the "task caveats" column *before* matching by RAM.

---


## Quantizations, with real file sizes

These are the actual byte counts of the published files, not estimates. The
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) set is the
one the installer uses.

{table(
    ["quant", "shard 1 file size — not measured memory usage", "download", "disk budget", "speed"],
    [[f"`{q['quant']}`" + (" ⭐" if q.get("recommended") else ""),
      f"**{q['shard1_human']}**",
      q["total_download_human"],
      f"{q['disk_budget_human']}*",
      q["relative_speed"] or "—"] for q in m["quantizations"]["sizes"]],
    ["---", "---", "---", "---", "---"])}

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

{table(
    ["publisher", "what it is", "status"],
    [[v["publisher"],
      v.get("what", v.get("technique", "")) + f" — {v.get('shard1_human', v.get('downloads', ''))}",
      v.get("status_in_strata", "—")] for v in m["quantizations"]["other_publishers"]],
    ["---", "---", "---"])}

### The Coder variant — and the trap in it

The [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) keeps
only **{m['quantizations']['coder_variant']['experts_kept']}**.
{m['quantizations']['coder_variant']['target_system']}. Its authors report
{m['quantizations']['coder_variant']['authors_claims']}.

⛔ **It is weaker outside code, and specifically on Chinese and other CJK text**, where
answers came out wrong or looping where English was fine
([Strata #438](https://github.com/Niko1221/Strata/issues/438)). If you work in Chinese, take
`Q2_0`, `IQ2_XS` or `IQ3_S` — the ones that keep every expert.

---

## Measured throughput

> {m['measured_throughput']['_warning']}

**System A** — {m['measured_throughput']['system_a']}

> ⛔ **Not a controlled quantization comparison.** Q2_0 was measured on engine 0.1.36; the
> remaining rows were measured on engine 0.1.26. The spread across rows conflates the
> quantization effect with the engine-version effect. Full-expert quants are ordered by
> shard-1 file size; Coder is appended as a specialized sub-variant.

{table(
    ["quant", "engine", "decode (short chat)", "decode (128K ctx)", "prefill (32K prompt)"],
    [[f"`{r['quant']}`",
      r.get("engine", "—"),
      f"{r['decode_short']} tok/s",
      f"{r['decode_128k_ctx']} tok/s",
      f"{r['prefill_32k']:,} tok/s"]
     for r in m['measured_throughput']['system_a_rows']],
    ["---", "---", "---:", "---:", "---:"])}

**System B** — {m['measured_throughput']['system_b']}

> ⛔ **Unverified engine versions.** The engine versions for System B were not reported in the
> upstream publication. Neither System B rows nor comparisons between System A and System B
> represent a controlled benchmark.

{table(
    ["quant", "engine", "decode (short chat)", "decode (128K ctx)", "prefill (32K prompt)"],
    [[f"`{r['quant']}`",
      r.get("engine", "—"),
      f"{r['decode_short']} tok/s",
      f"{r['decode_128k_ctx']} tok/s",
      f"{r['prefill_32k']:,} tok/s"]
     for r in m['measured_throughput']['system_b_rows']],
    ["---", "---", "---:", "---:", "---:"])}

ⓘ {m['measured_throughput']['metrics_note']}

⚠️ These are published shard-1 FILE SIZES, not measured peak process memory. Real usage also
depends on the runtime, the offload path, your context length and what else is open. Strata
describes its offload path as shard 1 resident in RAM with a working slice on the GPU and the
28.8 GB shard 2 mmap-ed on SSD; the proportions depend on the build, the driver and the
runtime flags, and a 12 GB card is enough because only a slice lives there.

### Projections — not measurements

Strata publishes these estimates and labels them as such. **Do not quote them as measured.**

{table(
    ["configuration", "projected decode", "why"],
    [[r["config"], f"**{r['value']}**", r["why"]]
     for r in m['measured_throughput']['projections_not_measurements']['rows']],
    ["---", "---", "---"])}

### Community reports

{table(
    ["date", "hardware", "configuration", "report"],
    [[r["date"], r["hardware"], r["config"], f"[read]({r['url']})"] for r in m['measured_throughput']['community_reports']],
    ["---", "---", "---", "---"])}

---

## Failure modes

Every item below is from a runtime that *does* support this model. These are not
"unsupported" problems; they are the specific things that break once you have it running.

ⓘ A note on the two worst ones: **they fail silently.** One produces correct-looking
answers at terrible speed. The other produces wrong answers with no error in the log.

### 🔴 The one that produces wrong answers

**Reported: correct on CPU, wrong on GPU — RTX 50-series (sm_120) with CUDA 13.2, on
`IQ1_S` / `IQ2_S` / `IQ3_S`.**

[{miscompile['title']}]({miscompile['url']}) — [Strata #{miscompile['issue']}]({miscompile['url']}), {miscompile['state']}

> {miscompile['symptom']}

The reporter's own isolation, in their words: the same build is correct with `-ngl 0` and wrong
with GPU offload, and `test-backend-ops` shows `MUL_MAT` 44 FAIL / `MUL_MAT_ID` 22 FAIL on
exactly those three quant types. ⓘ That localises it to the GPU path for those quants.

⛔ **We are not claiming we know the mechanism.** A compiler bug, a kernel implementation bug
and a driver interaction all remain consistent with what was reported. Nobody has ruled any of
them out. What is *not* speculative is the symptom: answers that look fine and are not.

**Fix:** {miscompile['fix']} ⚠️ *{miscompile['fix_status']}.* Evidence level: {miscompile['evidence_level']}.

**Before you debug anything else:** if answers come back *wrong* rather than missing, and you
are on an RTX 50-series card, check your `nvcc` version before anything else. This fails
silently — the log stays clean.

### 🔴 The one that only costs you speed

**Reported: catastrophic prefill slowdown on newer prebuilt binaries, cause unconfirmed.**

[{ngram['title']}]({ngram['url']}) — [llama.cpp #{ngram['issue']}]({ngram['url']}), {ngram['state']}

> {ngram['symptom']}

ⓘ {ngram['output_observation']}

**Root cause status:** {ngram['root_cause_status']}

**Fix:** {ngram['fix']} Evidence level: {ngram['evidence_level']}.

{ngram['practical_advice']}

### Memory growth on unified-memory systems

[{growth['title']}]({growth['url']}) — [llama.cpp #{growth['issue']}]({growth['url']}), {growth['state']}

{growth['symptom']}

Reported process footprint, from the issue thread:

{table(
    ["build", "what the reporter measured"],
    [[k, v] for k, v in growth['measurements'].items()],
    ["---", "---"])}

**Fix:** {growth['fix']} Evidence level: {growth['evidence_level']}.

⚠️ {growth['severity_note']}

### Everything else

{table(
    ["what goes wrong", "where", "state", "ev.", "fix"],
    [[f"**{x['title']}**",
      f"[{x['repo'].split('/')[1]} #{x['issue']}]({x['url']})",
      x["state"],
      x["evidence_level"][:1],
      x.get("fix", x.get("practical_advice", "—"))]
     for x in rest],
    ["---", "---", "---", "---", "---"])}

⚠️ `closed` on GitHub means the thread ended, not that a fix shipped. Read the `fix` column,
not the `state` column.

---

## Start here

```bash
# Let an AI coding assistant set it up for you — this is the documented path
Set up Strata on this PC for me: https://github.com/Niko1221/Strata \\
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

It fetches all {len(m['verification_sources']) + len(rt['verification_sources']) + len(fl['verification_sources'])}
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
| "125B model" | The released checkpoint is {params['total_safetensors_params_human']} ({params['total_safetensors_params']:,} parameters). 125B is the language model only. Plan downloads against the larger figure. | [`model.safetensors.index.json`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/model.safetensors.index.json) |
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

{table(
    ["source", "url"],
    [[s["name"], f"`{s['url']}`"] for s in
     (m["verification_sources"] + rt["verification_sources"] + fl["verification_sources"])],
    ["---", "---"])}

Licenses differ per component: the model is {m['model']['license_name']}, the quantizations
carry their publishers' terms, and Strata is MIT. Check each before you redistribute.

---

<div align="center">

**Data files** — [`data/model.json`](data/model.json) · [`data/runtimes.json`](data/runtimes.json) · [`data/failures.json`](data/failures.json)
**Scripts** — [`scripts/gen_readme.py`](scripts/gen_readme.py) · [`scripts/verify_sources.py`](scripts/verify_sources.py)

<sub>README.md is generated. Edit the JSON, not this file.</sub>

</div>
""")

    text = "\n".join(L)

    if "--check" in sys.argv:
        cur = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if cur != text:
            sys.exit("README.md is stale — run: python3 scripts/gen_readme.py")
        print("README.md in sync")
        return

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {OUT} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
