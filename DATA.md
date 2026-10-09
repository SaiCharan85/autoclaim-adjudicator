# Data sources

No data files are committed to this repository. Everything below is downloaded or generated locally
into the git-ignored `data/` folder. Ask before adding or changing any source.

| # | Source | Real? | Used for | License / terms | How to obtain |
|---|---|---|---|---|---|
| 1 | **NHTSA CRSS 2016–2024** (2022–24 = production pool; 2016–21 for the drift study, added 2026-10-04): nationally representative sample of US police-reported crashes ([downloads](https://static.nhtsa.gov/nhtsa/downloads/CRSS/)) | **real, recent** | Every claim's incident context; police-reported collision / animal / hit-and-run / parked / fire claims *are* these records | **Public domain** (U.S. Government work). Only `accident`, `vehicle`, `parkwork`, `violatn` files are kept; archive and file sha256 in `data/raw/crss_<year>/manifest.json`. Not redistributed. | `uv run python scripts/download_data.py crss_2016 ... crss_2024` (no account needed). Before 2019 the files carry codes only; names are filled from 2019–24 code maps (`lines/auto/crss.py`) |
| 2 | **Grounded-hybrid claims** (50,000 + two 8,000 holdouts) | real incidents + **simulated** policy, amounts, fraud, traps | Production fraud model; ground truth for harness evals | Ours (MIT). Built from source 1 + `config/simulator_auto.yaml`. See [docs/simulator.md](docs/simulator.md) | `uv run python scripts/simulate_claims.py` |
| 3 | Kaggle: *Vehicle Insurance Claim Fraud Detection* (15,420 rows, 1994–96) ([link](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)) | **real labels, old** | Reality check: the only real fraud labels publicly available | **CC0-1.0**, verified from Kaggle metadata on 2026-10-04 (sha256 in manifest). Not redistributed. | `uv run python scripts/download_data.py vehicle_fraud` (needs `KAGGLE_API_TOKEN`) |
| 4 | Synthetic claim narratives (300 eval from the test period, 300 dev from train + validation) | simulated | End-to-end harness evaluation (eval); prompt development, few-shot, feedback memory, fine-tuning data (dev) | Ours (MIT). Written by `gemini-3.5-flash-lite` (Google AI Studio free tier) from source 2's first-notice facts only (never labels or simulation truth). Not committed. | `uv run python scripts/build_narratives.py --set eval --n 300` and `--set dev` → `data/sim/packages/` |
| 5 | Personal auto policy text (75 clauses, 255 typed cross-references) + 60 labeled retrieval queries | ours | Coverage retrieval, policy graph, retrieval benchmark | Ours (MIT). Common US structure, our own wording; no ISO form text. | `src/autoclaim/lines/auto/policy/` |
| 6 | Fraud red-flag rules | ours | Rules engine in the fraud node | Ours (MIT), paraphrased from general industry knowledge; no NICB text. | `src/autoclaim/lines/auto/fraud_rules.yaml` |

| 7 | Embedding models `BAAI/bge-large-en-v1.5` (production), `bge-base`/`bge-small-en-v1.5` (benchmark reference), ONNX builds via fastembed | model weights | Dense policy retrieval (FAISS HNSW), local CPU only | **MIT** (BAAI). Downloaded once from Hugging Face to `.cache/fastembed/`; not redistributed | automatic on first use (`scripts/retrieval_benchmark.py` or the harness) |

| 8 | **NHTSA GES 2000–2015** (General Estimates System, CRSS's predecessor) ([listing](https://static.nhtsa.gov/?prefix=nhtsa/downloads/GES/)) | **real, older** | Row-level crash context for the 2000–2015 part of the multi-decade world (drift study) | **Public domain** (U.S. Government work). Not redistributed | Added 2026-10-04 (user approved). File names and formats vary by year (fixed-width 2000–08, flat/tab 2009–14, CSV 2015); make codes only |
| 9 | **BLS CPI** series `CUUR0000SETD` (motor vehicle maintenance & repair), `SETA01` (new vehicles), `SETA02` (used cars & trucks) | **real** | Era-correct amounts: repair costs and vehicle values per year | **Public domain**; citation requested ([BLS](https://www.bls.gov/opub/copyright-information.htm)) | BLS public API v1 (no key) or `download.bls.gov/pub/time.series/cu/` |
| 10 | **Cited single figures** (never copied tables/text): NAIC Auto Insurance Database Report; Insurance Information Institute auto facts (ISO/Verisk); CCC Crash Course; NICB press releases; state fraud-bureau annual reports (NY DFS, NJ OIFP, FL DFS, CA CDI, PA IFPA) | **real aggregates** | Calibrating claim frequency/severity, total-loss share and fraud mix by year | Copyright of each publisher; we record individual facts with source + page in `config/calibration_sources.yaml` | Added 2026-10-04 (user approved, "cite single figures only") |
| 11 | **US DOJ press releases** on auto-insurance fraud ([justice.gov/news](https://www.justice.gov/news)) | **real cases (text)** | Real fraud-scheme patterns for red-flag rules and narratives | Public domain unless marked ([DOJ](https://www.justice.gov/legalpolicies)) | Paraphrased only |
| 12 | **FEMA NFIP Redacted Claims** (OpenFEMA `FimaNfipClaims` v2), losses 2022-2025, 197,023 claims, coverage-relevant fields only (state is the only location kept) | **real** | Flood line plug-in (Step 8.5): deterministic flood payouts vs what was actually paid (`docs/flood_eval.md`); flood locked test = losses on or after 2024-07-01 | OpenFEMA terms (U.S. government data): cite FEMA with the required disclaimer; no re-identification; not to be used for determinations about anyone's rights or benefits ([terms](https://www.fema.gov/about/openfema/terms-conditions)) | `uv run python scripts/download_data.py fema_nfip` (free API, no key, ~20 paged requests) |
| 13 | **NHTSA vehicle-safety complaints** ([api.nhtsa.gov/complaints](https://api.nhtsa.gov/complaints/complaintsByVehicle?make=honda&model=civic&modelYear=2021)), crash reports only, 20 popular US models, model years 2021-2022 (~500 stories) | **real (text)** | Real-world statements of loss for the **Try a claim** page; owners' own words, used verbatim. Never used for training or evaluation (no truth labels) | **Public domain** (U.S. Government work, NHTSA Office of Defects Investigation). Not redistributed | `uv run python scripts/download_data.py nhtsa_complaints` (40 requests, no key; added 2026-10-08 at the user's request) |

## Checked and avoided (2026-10-04)
IIHS-HLDI bulk/API data (terms forbid derivatives), NICB member/NICTA/ForeCAST data, and anything
paid (ISO/Verisk Fast Track, full IRC studies).

## Checked and rejected (2026-10-04)
| Candidate | Why not |
|---|---|
| Kaggle `ahluwaliasaksham/car-insurance-fraud-detection-dataset` (2023–25, MIT) | Every column uniformly random and independent; only 16.7% of rows internally coherent |
| Kaggle `mmumairkhattak/insurance-claims-dataset-2026-fd-and-ra` (MIT) | Label leakage: `Fraud_Flag` = `Fraud_Risk_Score > 75` exactly; without leaked columns ROC-AUC 0.508; 75% non-auto; no dates |
| Kaggle `saisatish09/insuranceclaimsdata` (R `insuranceData` bundle) | Real but ≤2005, no fraud labels, license unknown |
| Kaggle `buntyshah/auto-insurance-claims-data` | ~1k rows, 2015, license unknown |
| Kaggle `arpan129/insurance-fraud-detection` | Copyright-authors license |

No real, recent, fraud-labelled auto-claims dataset is public (US, UK, EU and recent literature
checked). Recent research uses private insurer data.

## For later
- **FEMA NFIP redacted claims** (OpenFEMA, real, continuously updated, US government data): the
  planned real data source if a **flood** line module is added. It has no fraud labels.
- **CRSS 2025**: add it to `CRSS_YEARS` in `src/autoclaim/datasets/sources.py` once NHTSA publishes it.

## Notes
- Downloaded data stays local. Derived artifacts that could reconstruct it (row dumps, trained
  weights) are also git-ignored.
- Simulated fields and labels are fictional and must never be presented as real.


## FEMA disclaimer (required by the OpenFEMA terms)

This product uses the Federal Emergency Management Agency's OpenFEMA API, but is not endorsed by FEMA. The Federal Government or FEMA cannot vouch for the data or analyses derived from these data after the data have been retrieved from the Agency's website(s). Source: https://www.fema.gov/api/open/v2/FimaNfipClaims, accessed 2026-10-05. The flood line only replays historical, already-closed claims to test coverage logic; it makes no determination about any person.
