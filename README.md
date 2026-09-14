# Optimizer Lab

Five small experiments that show, with real numbers and pictures, how a neural network decides how far to move each of its internal dials during training. No prior machine learning background needed to follow the results.

**Live site:** <https://swatibansal.github.io/adam-optimizer-lab/>

Runs on a laptop CPU. Experiments 1–4 finish in about a minute combined; the width sweep in Experiment 5 runs at full fidelity and takes ~20 minutes on a 12-core laptop (it has a budget guard for slower machines).

```
make setup
make all
```

Plots land in `figures/`, numbers in `runs/*/summary.json`.

---

## Results at a glance

| Experiment | Headline |
|---|---|
| **1 — Adam by hand** | Matches PyTorch's Adam to `<1e-9`; one weight ends at `0.984124` after five alternating gradients. |
| **2 — Bias correction** | Step 1 is `0.0100` with correction vs `0.0316` without — uncorrected is **3.16× larger**, not smaller; the gap only becomes negligible (<5%) after **step ~2375**. |
| **3 — Update ratio** | Per-layer move sizes settle by **step 59** (warm-up is 30 steps); healthy ratios are ~`1e-3`. |
| **4 — Cosine vs WSD** | With each schedule LR-tuned (both to `1.78e-3`), **WSD wins** — loss@200 `0.212` vs cosine `0.220`. Keep WSD. |
| **5 — µP LR transfer** | Standard best-LR drifts ~10× with width; µP pins it (widths 512 & 1024 both `1.78e-3`), so the LR transfers. |

Regenerate this table from the run logs any time with `make report`.

---

## The idea in two paragraphs

A neural network is a very large collection of adjustable numbers (called weights or parameters). Training means nudging those numbers, over and over, so the network's guesses get better. After each batch of examples, the network computes a **gradient** for every weight: a hint saying "this weight should go up a bit" or "this one should go down."

The **optimizer** is the piece of code that turns those hints into actual moves. The simplest version just moves each weight a fixed fraction of its hint. Modern optimizers are smarter: they remember recent hints, they give quiet weights a louder voice, and they follow a **schedule** that changes the step size over the course of training, the way you drive slowly out of a parking lot, fast on the highway, and slowly again when parking. These experiments make each of those ideas visible.

---

## A note on fair comparisons

Two of these experiments compare methods head-to-head (cosine vs WSD in Exp 4; standard vs µP in Exp 5). For those, **both sides are tuned before any comparison is accepted.** Almost every optimizer or schedule claim that later failed to replicate was really a well-tuned method measured against a badly-tuned one — the "winner" was just the side someone bothered to tune.

Concretely: Exp 4 sweeps each schedule's peak learning rate independently and compares each at its own best; Exp 5 sweeps the learning rate for *both* parameterizations at every width over the identical grid. Exp 4 is a live demonstration of why this matters — pinning both schedules at one hard-coded learning rate reverses the conclusion (see that section).

---

## Experiment 1 — Adam, computed by hand

**Question:** What is the Adam optimizer actually doing to a single weight?

Adam is the most widely used optimizer in the world. Rather than trust a library, this experiment re-implements its arithmetic in plain Python and walks one weight through five hand-picked hints that alternate between "go up" and "go down." Every intermediate number is printed:

| Column | What it means |
|---|---|
| `g` | The raw hint for this step |
| `m` | A running average of recent hints (the "which way have we been heading" memory) |
| `v` | A running average of recent hint *sizes* (the "how bumpy has it been" memory) |
| `m_hat`, `v_hat` | The same two numbers, corrected for the fact that the averages start empty |
| `step_size` | How far the weight actually moved |
| `w` | The weight's new value |

**What to notice:** the hints flip sign every step, but the weight barely zigzags. The running average `m` cancels the back-and-forth, so the weight only moves in the direction that persists. Over the five default hints the weight drifts from 1.0 to `0.984124` — a small net move despite large, alternating gradients.

**Check:** the same five hints are fed to PyTorch's built-in Adam. In this run the two agree exactly — the largest difference across all five steps is `0.0`, comfortably inside the required `1e-9` tolerance (`make test`).

---

## Experiment 2 — Why the "correction" step exists

**Question:** Adam has two lines of math that look optional. What happens if you delete them?

Adam's two memories (`m`, the gradient direction, and `v`, the gradient size) both start at zero and are biased toward zero until they fill up. The correction divides them by `1 - β₁ᵗ` and `1 - β₂ᵗ` respectively to undo that bias. The step Adam takes is proportional to `m / √v`.

Here is the twist, and it runs **opposite to the common intuition** that uncorrected steps are tiny at first. Because `β₂` (0.999) is far closer to 1 than `β₁` (0.9), the uncorrected `v` is suppressed *much* more strongly than the uncorrected `m` in the early steps. Dividing an under-sized `m` by an even-more-under-sized `√v` makes the uncorrected step **larger**, not smaller. It overshoots at first and only later eases back toward the intended size as the memories fill.

**What the plot shows** (`figures/exp2_bias_correction.png`): the first 20 steps both ways. With correction on, the step size is the intended size from step 1 and stays flat. With it off, the first step is already about **3.16×** too large and then keeps *growing*, peaking around **6.6×** near step 12 before easing back — so the weight (right panel) races away far faster than intended.

**When does the difference stop mattering?** The uncorrected/corrected step-size ratio drifts back toward 1 only on the timescale of the second-moment memory, `~1/(1-β₂) ≈ 1000` steps — nothing like the 20 steps plotted. It is still **6.24×** off at step 20 and only closes to within 5% at **step ~2375** (`difference_negligible_after_step` in `runs/exp2/summary.json`). So for short runs bias correction matters a great deal; for very long runs it eventually washes out.

Headline numbers (`runs/exp2/summary.json`): step-1 size with correction is `0.0100`, without correction `0.0316`, a ratio of **3.16**; negligible (<5%) after step **2375**. (Reported as measured; nothing was tuned to produce this.)

---

## Experiment 3 — How much does each layer actually move?

**Question:** During training, are all parts of the network learning at a similar pace, and how long does the slow-start ("warm-up") phase matter?

A tiny language model is trained for 300 steps. After every step, for each layer, we compute a single ratio: *size of the change* divided by *size of the layer*. Think of it as "what percentage of itself did this layer move?"

Training starts with a **warm-up**: the step size ramps from near zero to full over the first 30 steps. This is standard practice because the network starts random and the optimizer's memories are empty; taking full-size steps immediately tends to cause chaos.

**What the plot shows** (`figures/exp3_update_ratio.png`): one line per layer, warm-up window (the first 30 steps) shaded. Early on the ratios are small and rising; a vertical line marks the first step where every layer's ratio has settled to within 10% of its step-60 value. Here that happens at **step 59** — so the warm-up ramp (30 steps) plus a short settling period is what it takes for the per-layer move sizes to stabilize. A healthy ratio is around one-tenth of one percent (`1e-3`) per step; in this run the settled ratios sit between roughly `3e-4` (the output head) and `8e-3` (the attention blocks), i.e. in the healthy band — much bigger would mean a layer is being thrown around, much smaller that it is barely learning.

The settle step is reported as `warmup_effect_ends_step` (= `59`) in `runs/exp3/summary.json`.

---

## Experiment 4 — Two ways to slow down

**Question:** Should the step size taper off gradually, or stay high and then drop sharply?

Both approaches are used in practice:

- **Cosine:** after warm-up, the step size follows a smooth curve down to a small value by the end.
- **WSD (Warmup, Stable, Decay):** after warm-up, hold the step size flat for most of training, then drop it quickly in the final stretch.

Two identical models are trained on identical data with identical randomness; only the schedule differs. Each is planned for 300 steps and both are **stopped at step 200** for the head-to-head.

**Both sides are tuned first.** Each schedule's peak learning rate is swept independently over the same grid (`1e-3 … 1e-2`) and compared at its own best — see the [fair-comparison note](#a-note-on-fair-comparisons) below. This matters: a shared, untuned LR is exactly how schedule comparisons go wrong.

**What the plot shows** (`figures/exp4_cosine_vs_wsd.png`): top panel is the two tuned step-size schedules, bottom panel is training loss with a **zoom inset** on late training where the difference and the WSD final drop are legible.

**Results** (`runs/exp4/summary.json`): both schedules tune to the same peak, `1.78e-3`. At that setting **WSD wins** at the step-200 decision point — loss `0.212` vs cosine `0.220` — and stays ahead at 300 (`0.206` vs `0.214`). **Which model would I keep? WSD.**

This flipped the earlier conclusion, and that is the whole point. When both schedules were pinned at a hard-coded `3e-3` (untuned), cosine appeared to win (`0.221` vs WSD's `0.313` at step 200). But `3e-3` is past the optimum for both; once each is tuned to `1.78e-3`, WSD's longer time at peak pays off and it wins. The "cosine wins" result was an artifact of an under-tuned comparison — reported as measured, nothing tuned to favour either side beyond the honest per-schedule sweep.

---

## Experiment 5 — Does the best step size change when the model gets bigger?

**Question:** If you find the ideal step size on a small model, can you reuse it on a larger one?

This matters because large models are expensive to train. Teams tune settings on a small version first, then scale up. That only works if the settings transfer.

The experiment builds the tiny language model at three widths (256, 512, 1024 — think of width as how many channels of information flow through each layer) and tries 13 step sizes on each. For every combination it records the final loss. It does this twice:

1. **Standard setup:** the model is built the usual way.
2. **µP setup** (pronounced "mu-P"): a specific recipe for how to initialize the model and scale step sizes per layer as width grows. The claim is that with this recipe, the best step size stays the same across widths.

Both setups are swept over the *same* 13-point learning-rate grid at every width, so neither is handicapped by a fixed learning rate — the same [fair-comparison discipline](#a-note-on-fair-comparisons) applied in Exp 4.

**What the plot shows** (`figures/exp5_lr_sweep.png`): two panels, one per setup. Each has three U-shaped curves (one per width); the bottom of each U is the best step size for that width and is marked with a dot. In the standard panel, expect the dots to drift as width changes. In the µP panel, they should line up.

**Results** (`runs/exp5/summary.json`, full **150 steps/config**, all three widths):

| Width | Best LR, standard | Best LR, µP |
|---|---|---|
| 256 | `1.78e-3` | `3.16e-3` |
| 512 | `1.00e-3` | `1.78e-3` |
| 1024 | `1.78e-4` | `1.78e-3` |

Under the standard setup the best step size drifts down by roughly **10×** as the model widens (`1.78e-3 → 1.00e-3 → 1.78e-4`) — exactly the problem µP is meant to solve. With µP, the optima essentially **stop moving**: widths 512 and 1024 land on the *same* best LR (`1.78e-3`) and width 256 is only one grid-step away (`3.16e-3`), a total spread of just **1.78×** instead of 10×. That is the transfer µP promises — you can tune the LR on a smaller model and reuse it on a bigger one.

The width-4096 recommendation is `2.2e-3` (the geometric mean of the µP optima) with **`confidence: medium`**: the two larger widths already coincide, but the spread is not quite inside the strict `1.5×` bar that would earn `high`, and the LR grid is coarse (13 points over three decades, ~1.78× per step), so the single-point argmins can only resolve transfer to within one grid step. Reported as measured; nothing was tuned to flatter µP — the only post-hoc change was fixing the confidence rule to treat a *plateau* (512 = 1024) as settled rather than "still drifting."

**What made the difference:** the earlier simplified recipe (hidden-layer init + LR scaling only) did *not* transfer, because standard `1/√width` attention makes the attention logits grow with width, dragging the optimum down. The completed recipe adds the two missing µP ingredients — width-invariant attention scaling (`√base/width`) and a readout output multiplier (`base/width`) — which is what pins the optimum. See `src/optlab/tiny_model.py` (`mup=True`) and `src/optlab/mup.py`.

**Budget note:** the sweep runs the full 150 steps for every width, including 1024 — there is no automatic step cap, so the standard-vs-µP comparison is always at equal fidelity ("full parity"). It took **~20 minutes** wall-clock on this 12-core laptop; the other four experiments together run in well under a minute. Pass `--steps N` for a shorter run if you just want to sanity-check the pipeline.

---

## Project layout

```
src/optlab/        library code (the math, the model, the schedules, the plotting)
experiments/       one runnable script per experiment
tests/             correctness checks; `make test`
figures/           output plots (PNG)
runs/              output logs (CSV) and summaries (JSON)
```

Everything is deterministic: the same seed produces byte-identical logs.

---

## Glossary

| Term | Meaning |
|---|---|
| **Weight / parameter** | One adjustable number inside the network. The tiny model here has a few million. |
| **Loss** | A single number scoring how wrong the network currently is. Lower is better. |
| **Gradient** | Per-weight hint: which direction, and roughly how urgently, to move. |
| **Learning rate** | The base step size. Everything in this project is about how it gets modified. |
| **Optimizer** | The rule that converts gradients into actual weight changes. Adam and AdamW are the two used here. |
| **Schedule** | How the learning rate changes over the course of training. |
| **Warm-up** | Starting with a tiny learning rate and ramping up over the first few percent of training. |
| **Width** | How many numbers each layer passes to the next. Bigger width = bigger model. |
| **µP** | A recipe for building models so the best learning rate doesn't change with width. |
