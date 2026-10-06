# Flood line (Step 8.5): a second line of business on the same harness

**What it shows:** the shared core (graph, router, retries, audit, budgets, idempotent
finalization, human review, feedback memory) runs a different insurance line with **no core
changes**. Everything flood-specific lives in `src/autoclaim/lines/flood/`; a test checks that
`core/` contains no flood code.

**Data:** real FEMA NFIP redacted flood claims, losses 2022-2025 (197,023 claims; `DATA.md` #12).
Only coverage-relevant fields and the state are kept: no coordinates, census or ZIP fields.

## How a flood claim is decided (code decides every number)

Plain-English trace of a real record: tropical storm (tidal overflow) -> building damage $6,948
-> building deductible code "2" = $2,000 -> $4,948 payable, under the $250,000 limit ->
auto-approved. FEMA's record shows $4,947.64 paid.

| step | rule (clause) |
|---|---|
| each part separately | building and contents each pay damage minus their own deductible, capped at their limit (`FLD-DEDUCTIBLE`, `FLD-LIMITS`) |
| nothing above the deductibles | deny (`FLD-BELOW-DEDUCTIBLE`) |
| judgment needed | escalate: cause not clearly a flood (`FLD-EXC-NOT-FLOOD`), no damage documented or a placeholder amount (`FLD-COND-PROOF`), unknown deductible, a loss dated before the record's policy start (check the policy history) |
| large payments | over the $50,000 authority limit -> adjuster (`config/carrier_config.yaml: flood.router`) |

The rules were fixed from losses before 2024-07-01 only (`lines/flood/claim.py` lists each one
with the evidence behind it); the later period was then scored once (`--final`, logged).

Two findings from the real data that shaped the rules:
- A blank damage amount means that part was not claimed (only 47 of 15,303 blank contents
  amounts were paid), but an amount above $0 and under $10 is a placeholder (seen with payments up
  to $250,000), so it goes to an adjuster.
- The record's policy start date is not the true inception for renewals or transfers: most losses
  dated before it were paid, so they escalate instead of being denied.

## Results (template mode: no LLM calls, $0)

| metric | development (losses < 2024-07-01) | **locked test** (>= 2024-07-01) |
|---|---|---|
| decided without an adjuster | 45.4% | 41.4% |
| auto-decisions agreeing with what FEMA did | 97.6% | **98.6%** |
| wrongly denied (FEMA paid) | 0.6% | 0.3% |
| wrongly paid (FEMA paid nothing) | 0.5% | 0.3% |
| actual-cash-value payouts within 5% of FEMA's | 82.3% | 87.3% |
| replacement-cost claims: our first payment / final total | 0.932 | 0.937 |

3,000 claims sampled per period (seed 42), 95% bootstrap CIs in `docs/flood_eval.md` and
`docs/flood_eval_test.md`. Replacement-cost claims pay the actual cash value first and the rest
after repairs, which is why our figure is ~93% of the final total.

Most escalations are the authority limit (big hurricane losses), not uncertainty: the cases that
need judgment are about 22% (development) and 12% (locked test) of claims.

**LLM mode** (`FloodLine(client=..., judge_impl=...)`): the adjudicator LLM writes the explanation
around the code's numbers and JudgeKit grades it; nothing else changes. Not run at scale (free
quota is reserved for the auto evaluation).

_This product uses the Federal Emergency Management Agency's OpenFEMA API, but is not endorsed by
FEMA. The Federal Government or FEMA cannot vouch for the data or analyses derived from these data
after the data have been retrieved from the Agency's website(s)._ Historical, closed claims only;
no determination about any person is made.
