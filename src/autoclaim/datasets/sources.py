"""Registry of external datasets. See DATA.md for licenses and terms."""

from pydantic import BaseModel, ConfigDict


class DatasetSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    kaggle_ref: str  # "owner/dataset-slug"
    files: tuple[str, ...]  # files we expect after unzipping
    description: str


VEHICLE_FRAUD = DatasetSource(
    key="vehicle_fraud",
    kaggle_ref="shivamb/vehicle-claim-fraud-detection",
    files=("fraud_oracle.csv",),
    description="Vehicle Insurance Claim Fraud Detection (~15k claims, FraudFound_P label)",
)

SOURCES: dict[str, DatasetSource] = {s.key: s for s in (VEHICLE_FRAUD,)}
