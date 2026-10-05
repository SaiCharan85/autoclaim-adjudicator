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
    optional_files: tuple[str, ...] = ()  # kept when present (e.g. value-label files)


VEHICLE_FRAUD = DatasetSource(
    key="vehicle_fraud",
    kaggle_ref="shivamb/vehicle-claim-fraud-detection",
    files=("fraud_oracle.csv",),
    description="Vehicle Insurance Claim Fraud Detection (~15k claims, FraudFound_P label)",
)

# NHTSA Crash Report Sampling System: nationally representative sample of US police-reported
# crashes. Only the crash, vehicle, struck-parked-vehicle and violation files are kept.
CRSS_YEARS = (2022, 2023, 2024)  # the production simulator's incident pool
CRSS_ALL_YEARS = tuple(range(2016, 2025))  # CRSS began in 2016 (GES before); for the drift study
CRSS_FILES = ("accident.csv", "vehicle.csv", "parkwork.csv", "violatn.csv")
CRSS: dict[int, UrlSource] = {
    year: UrlSource(
        key=f"crss_{year}",
        url=f"https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip",
        files=CRSS_FILES,
        license="Public domain (U.S. Government work, NHTSA)",
        description=f"NHTSA CRSS {year}: police-reported crashes (crash/vehicle/parked/violation)",
    )
    for year in CRSS_ALL_YEARS
}

# NHTSA GES (CRSS's predecessor), SAS releases. 2000-2001 ship only SAS-6 .sd2 files that no
# Python reader handles, so the series starts in 2002. Value labels come from formats.sas7bcat or
# PROC FORMAT source (2002, 2005, 2012); 2006 has neither (labels borrowed: lines/auto/ges.py).
GES_BASE = "https://static.nhtsa.gov/nhtsa/downloads/GES"
GES_ARCHIVES = {
    2002: "GES02/SAS/GES02.zip", 2003: "GES03/SAS/GES03.zip", 2004: "GES04/GES04.zip",
    2005: "GES05/SAS/GES2005.zip", 2006: "GES06/SAS/GES2006.zip", 2007: "GES07/GES2007.zip",
    2008: "GES08/GES2008.zip", 2009: "GES09/GES09%20PCSAS.zip", 2010: "GES10/GES10_PCSAS.zip",
    2011: "GES11/GES11_PCSAS.zip", 2012: "GES12/GES12_PCSAS.zip", 2013: "GES13/GES13_PCSAS.zip",
    2014: "GES14/GES2014SAS.zip", 2015: "GES15/GES2015sas.zip",
}  # fmt: skip
GES_LABEL_FILES = ("formats.sas7bcat", "Format02.sas", "Format05_lookalike.sas", "Format12.sas")
GES: dict[int, UrlSource] = {
    year: UrlSource(
        key=f"ges_{year}",
        url=f"{GES_BASE}/{path}",
        files=("accident.sas7bdat", "vehicle.sas7bdat", "violatn.sas7bdat"),
        optional_files=("parked.sas7bdat", "parkwork.sas7bdat", *GES_LABEL_FILES),
        license="Public domain (U.S. Government work, NHTSA)",
        description=f"NHTSA GES {year}: police-reported crashes (SAS release)",
    )
    for year, path in GES_ARCHIVES.items()
}

SOURCES: dict[str, DatasetSource] = {s.key: s for s in (VEHICLE_FRAUD,)}
URL_SOURCES: dict[str, UrlSource] = {s.key: s for s in (*CRSS.values(), *GES.values())}
