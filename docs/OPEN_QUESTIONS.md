# Open Questions

Working notes on unresolved methodology questions for the OOD generalization
work. Answered where the evidence supports it; flagged for follow-up where
it doesn't.

---

## 1. Does the original method test for over-refusal, and if not, which of the other 7 methods do?

**The observation**: Safety Layers' *final* comparison (Section 4's R_h/S_h
table, SPPFT vs. Full FT) only tests on AdvBench -- 520 prompts that are
100% unambiguously harmful by construction. There's no paired "safe"
prompt set in that final comparison, so the paper's headline result can't
by itself tell you whether SPPFT's safety layers also cause the model to
start refusing benign requests (over-refusal / false positives).

**Correction worth noting**: Safety Layers *does* have an over-refusal
dataset in the original repo -- `Dataset/Evaluation/Over_rejection_dataset.csv`
(731 borderline-but-benign prompts, our copy at `data/over_rejection.csv`).
It's just used earlier, in Section 3.4's layer-*scaling* validation (does
scaling the identified safety-layer range increase over-rejection?), and
never carried into Section 4's actual SPPFT-vs-FullFT R_h/S_h comparison.
So the gap is real, but it's a gap in what got *carried forward* into the
final result, not a total absence of an over-refusal dataset in the
project.

**What the other 7 methods (Table 1) actually do here**, from the
method-comparison table already built this session:

| Method | Has an explicit safe/benign-prompt check? | Dataset used |
|---|---|---|
| Safety-Specific Neuron (Zhao) | No explicit over-refusal set found | -- |
| Safety Neurons (Chen) | Utility check only (not over-refusal specifically) | MMLU, GSM8K, BBH, TruthfulQA |
| SAFEx (Lai) | Ambiguous -- `WildGuard` (part of `D_regular`) contains some benign examples as a moderation-training set, but this isn't framed as a dedicated over-refusal test | WildGuard (partial) |
| **Refusal Direction (Arditi)** | **Yes** -- explicitly | 100 harmless prompts from **Alpaca**, used in the "addition" arm of their evaluation: adding the refusal direction to *harmless* prompts and checking whether it induces refusal, to demonstrate the direction is causally bidirectional |
| Universal Refusal Direction (Wang) | Same Alpaca-based harmless contrast at the direction-finding stage as Arditi; not clearly re-applied at the cross-lingual attack stage | Alpaca |
| **Representation Bending (Yousefpour)** | **Yes, explicitly and directly** | **XSTest** -- a benchmark purpose-built to test exaggerated safety / over-refusal on safe-but-scary-sounding prompts |
| Safety Fine-Tuning (Jain) | Indirectly, via construction: the synthetic PCFG data pairs a harmful concept with a structurally-identical benign one (e.g. "design a bomb" vs. "design a bike"), which functions as a built-in over-refusal control | Self-generated synthetic pairs, not a separate named dataset |
| Safety Layers (Li) | Yes, but not in the final comparison (see above) | `over_rejection.csv`, 731 prompts -- used in Sec. 3.4 only |

**Practical takeaway for our own OOD work**: if Safety Layers' OOD
evaluation is going to be a fair, complete picture, it probably shouldn't
stop at "does R_h go up on OOD content" -- it should also ask "does the
model start refusing benign OOD-adjacent content it wouldn't have refused
before." We already have a natural candidate for this: the **ambiguous-
label subset** flagged in the curation tool (prompts whose own source
dataset gave inconsistent safety verdicts) is a reasonable proxy for
"borderline, not clearly deserving refusal" -- which is functionally close
to what XSTest is designed to test, just derived differently. The
stratified analysis (`scripts/analyze_ood_eval_result.py`, "Option A")
already reports S_h/R_h separately for that subset, which is a lightweight
version of the over-refusal check the other methods build in more
formally. Worth considering `over_rejection.csv` itself (Safety Layers'
own 731-prompt over-refusal set) as a *third* comparison in the final
write-up, alongside AdvBench (ID) and the new OOD set -- since it's
already sitting in the repo unused for this purpose.

---

## 2. One dataset's evaluation methodology requires human judgment of generated output -- how do we handle that?

**Status: need clarification before this can be answered properly.**

I don't have a confident, verified match for which specific dataset/row
you mean -- I don't want to guess and give you an answer anchored to the
wrong entry. Could you point to which one (method-comparison table row,
or a Table 2/3 dataset) has this property? Once identified, I'll verify
its actual evaluation methodology from the source paper before answering,
same as the rest of this project's approach.

**In the meantime, the general framework this project has already been
using for exactly this situation** (a benchmark whose original
methodology calls for human evaluation of generated responses):

1. **Substitute a strong LLM judge as a documented proxy** -- this is
   literally what `harmful_score.py` already does for S_h: Qi et al.
   2023's methodology, originally scored by human/GPT-4 judges, is
   reproduced here with Claude Haiku 4.5 as the judge, explicitly
   documented as a substitution rather than silently treated as
   equivalent.
2. **Check if the paper also published an automated classifier** as an
   alternative/validation against their human ratings -- if so, prefer
   that over building a new proxy from scratch (this is why HarmBench's
   own official classifier is used here rather than only the Zou keyword
   heuristic).
3. **If neither exists**, the honest options are: (a) run a small, real
   human evaluation ourselves and document the sample size/rater
   agreement, or (b) exclude that specific dataset from quantitative
   head-to-head comparison and cite it only qualitatively, being explicit
   about why -- rather than quietly approximating it with a method that
   wasn't validated against it.

Which option makes sense depends on the specific dataset -- hence needing
to know which one first.
