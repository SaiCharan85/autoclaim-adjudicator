# Limitations

What the results in the README do and do not show.

## The data

- **Auto claims are simulated.** Each claim is grounded in a real police-reported crash (NHTSA CRSS
  2022-2024) and priced with real price indexes, but the policy, the claimant's story, the fraud and
  the "right" decision come from a simulator ([simulator.md](simulator.md)). The harness is measured
  against that simulator's truth, which encodes our reading of a fictional policy, not an insurer's
  claim files.
- **Narratives are LLM-written** from the simulated facts (with style cards and deliberate omissions),
  so they are cleaner and more uniform than real first notices of loss.
- **Fraud labels are synthetic.** True fraud and the share an insurer would confirm are simulated;
  the triage numbers say how well the models find the simulator's fraud patterns, not real fraud.
  The real 1990s Kaggle dataset is used only as an honesty check, and scores much lower.
- **The flood line uses real outcomes** (FEMA NFIP, 2022-2025), but it measures deterministic payout
  math against what was paid. It does not test LLM judgment, and FEMA's redacted records lack the
  adjuster's notes that decide many escalations.

## The models and judges

- **Judge and fine-tuning results come from synthetic tests.** Planted errors are edits to
  correct-by-construction template explanations. The fine-tuned judge was trained and tested on that
  same kind of data (95% catch rate), so it will do worse on real adjudicator explanations, which are
  worded differently. The pilot with an API judge on 10 clean + 44 planted cases is too small for
  firm conclusions (its 20% false-alarm rate rests on 2 of 10 cases).
- **The simulated adjuster is the truth with 5% noise.** Real adjusters disagree with each other far
  more, so feedback memory is tested in an easy setting.
- **Adjudicator distillation was deferred**: only the intake extractor and the judge were fine-tuned.

## Scale and cost

- **Free tiers cap throughput** at about 40-50 harness claims per day, so the evaluations are pilots
  (tens of claims, wide confidence intervals) rather than thousands of claims. Free-tier models also
  change and get rate-limited without notice; one model returned quota errors for a whole day.
- **The local fallback is slow** on a CPU (about 17-34 seconds per call): fine for quota-out days,
  not for volume.

## Scope

- Auto physical damage only (collision and comprehensive) under one fictional personal auto policy,
  US rules. No liability, bodily injury, subrogation, salvage or litigation.
- One flood policy, building and contents only.
- Decisions are recommendations from a research system. Nothing here should decide a real person's
  claim; the FEMA data terms forbid using that data for such determinations.
