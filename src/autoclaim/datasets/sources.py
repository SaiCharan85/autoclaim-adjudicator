"""Registry of external datasets. See DATA.md for licenses and terms."""

from pydantic import BaseModel, ConfigDict


class DatasetSource(BaseModel):
    """A Kaggle dataset."""

    model_config = ConfigDict(frozen=True)

    key: str
    kaggle_ref: str  # "owner/dataset-slug"
    files: tuple[str, ...]  # files we expect after unzipping
    description: str


class UrlSource(BaseModel):
    """A zip archive at a public URL; only `files` (matched by basename) are extracted."""

    model_config = ConfigDict(frozen=True)

    key: str
    url: str
    files: tuple[str, ...]
    license: str
    description: str


VEHICLE_FRAUD = DatasetSource(
    key="vehicle_fraud",
    kaggle_ref="shivamb/vehicle-claim-fraud-detection",
    files=("fraud_oracle.csv",),
    description="Vehicle Insurance Claim Fraud Detection (~15k claims, FraudFound_P label)",
)

# NHTSA Crash Report Sampling System: nationally representative sample of US police-reported
# crashes. Only the crash, vehicle, struck-parked-vehicle and violation files are kept.
CRSS_YEARS = (2022, 2023, 2024)
CRSS_FILES = ("accident.csv", "vehicle.csv", "parkwork.csv", "violatn.csv")
CRSS: dict[int, UrlSource] = {
    year: UrlSource(
        key=f"crss_{year}",
        url=f"https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip",
        files=CRSS_FILES,
        license="Public domain (U.S. Government work, NHTSA)",
        description=f"NHTSA CRSS {year}: police-reported crashes (crash/vehicle/parked/violation)",
    )
    for year in CRSS_YEARS
}

SOURCES: dict[str, DatasetSource] = {s.key: s for s in (VEHICLE_FRAUD,)}
URL_SOURCES: dict[str, UrlSource] = {s.key: s for s in CRSS.values()}
