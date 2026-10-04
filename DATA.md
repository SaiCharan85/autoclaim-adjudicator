# Data sources

No data files are committed to this repository. Everything below is downloaded or generated locally
into the git-ignored `data/` folder. Ask before adding or changing any source.

| # | Source | Real? | Used for | License / terms | How to obtain |
|---|---|---|---|---|---|
| 1 | **NHTSA CRSS 2022, 2023, 2024**: nationally representative sample of US police-reported crashes ([downloads](https://static.nhtsa.gov/nhtsa/downloads/CRSS/)) | **real, recent** | Every claim's incident context; police-reported collision / animal / hit-and-run / parked / fire claims *are* these records | **Public domain** (U.S. Government work). Only `accident`, `vehicle`, `parkwork`, `violatn` files are kept; archive and file sha256 in `data/raw/crss_<year>/manifest.json`. Not redistributed. | `uv run python scripts/download_data.py crss_2022 crss_2023 crss_2024` (no account needed) |
| 2 | **Grounded-hybrid claims** (50,000 + two 8,000 holdouts) | real incidents + **simulated** policy, amounts, fraud, traps | Production fraud model; ground truth for harness evals | Ours (MIT). Built from source 1 + `config/simulator_auto.yaml`. See [docs/simulator.md](docs/simulator.md) | `uv run python scripts/simulate_claims.py` |
| 3 | Kaggle: *Vehicle Insurance Claim Fraud Detection* (15,420 rows, 1994–96) ([link](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)) | **real labels, old** | Reality check: the only real fraud labels publicly available | **CC0-1.0**, verified from Kaggle metadata on 2026-10-04 (sha256 in manifest). Not redistributed. | `uv run python scripts/download_data.py vehicle_fraud` (needs `KAGGLE_API_TOKEN`) |
| 4 | Synthetic claim narratives | simulated | End-to-end harness evaluation | Ours (MIT). LLM-written from source 2 rows (Step 4). | `uv run python scripts/generate_claims.py` (Step 4) |
| 5 | Personal auto policy text | ours | Coverage retrieval and policy graph | Ours (MIT). Common US/international structure, our own wording; no ISO form text. | `policy/` (Step 3) |
| 6 | Fraud red-flag rules | ours | Rules engine in the fraud node | Ours (MIT), paraphrased from general industry knowledge; no NICB text. | `src/autoclaim/lines/auto/fraud_rules.yaml` |

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
