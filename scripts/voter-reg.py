"""Load CPS fixed-width voter registration extract; output registration-only microdata CSV."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Layout for abbreviated IPUMS CPS extract (abrv_voter_reg.dat): 1-based inclusive positions.
SPECS = {
    "YEAR": (1, 4),
    "STATEFIP": (50, 51),
    "METFIPS": (59, 63),
    "FAMINC": (70, 72),
    "AGE": (129, 130),
    "SEX": (131, 131),
    "RACE": (132, 134),
    "VOREG": (157, 158),
    "WTFINL": (75, 88),
}

STATEFIP_LABELS = {
    1: "Alabama",
    2: "Alaska",
    4: "Arizona",
    5: "Arkansas",
    6: "California",
    8: "Colorado",
    9: "Connecticut",
    10: "Delaware",
    11: "District of Columbia",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    21: "Kentucky",
    22: "Louisiana",
    23: "Maine",
    24: "Maryland",
    25: "Massachusetts",
    26: "Michigan",
    27: "Minnesota",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    31: "Nebraska",
    32: "Nevada",
    33: "New Hampshire",
    34: "New Jersey",
    35: "New Mexico",
    36: "New York",
    37: "North Carolina",
    38: "North Dakota",
    39: "Ohio",
    40: "Oklahoma",
    41: "Oregon",
    42: "Pennsylvania",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    47: "Tennessee",
    48: "Texas",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    53: "Washington",
    54: "West Virginia",
    55: "Wisconsin",
    56: "Wyoming",
}

SEX_LABELS = {1: "Male", 2: "Female"}


def _build_colspecs(specs):
    names = list(specs.keys())
    colspecs = [(s - 1, e) for s, e in specs.values()]
    return colspecs, names


def _resolve_data_file() -> Path:
    candidates = [
        PROJECT_ROOT / "data/raw/voter_registration/abrv_voter_reg.dat",
        PROJECT_ROOT / "data/files/abrv_voter_reg.dat",
        PROJECT_ROOT / "data/processed/voter_reg/abrv_voter_reg.dat",
        PROJECT_ROOT / "abrv_voter_reg.dat",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Place abrv_voter_reg.dat under data/raw/voter_registration/ "
        "(or project root)."
    )


def load_raw_dataframe():
    colspecs, names = _build_colspecs(SPECS)
    data_path = _resolve_data_file()
    return pd.read_fwf(data_path, colspecs=colspecs, names=names, dtype=str)


def clean_for_modeling(df):
    numeric_cols = [
        "YEAR",
        "STATEFIP",
        "METFIPS",
        "FAMINC",
        "AGE",
        "SEX",
        "RACE",
        "VOREG",
        "WTFINL",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["VOREG"].isin([96, 97, 98, 99]), "VOREG"] = pd.NA
    df.loc[df["FAMINC"].isin([995, 996, 997, 999]), "FAMINC"] = pd.NA
    df = df.loc[df["AGE"] >= 18].copy()
    df["voter_registration"] = df["VOREG"].map({1: 0, 2: 1})
    df["state"] = df["STATEFIP"].map(STATEFIP_LABELS)
    df["sex_label"] = df["SEX"].map(SEX_LABELS)
    df["faminc_code"] = df["FAMINC"]

    keep_cols = [
        "YEAR",
        "state",
        "STATEFIP",
        "METFIPS",
        "AGE",
        "sex_label",
        "RACE",
        "faminc_code",
        "voter_registration",
        "WTFINL",
    ]
    return df[keep_cols]


def create_individual_response_subset(df):
    subset = df.rename(
        columns={
            "YEAR": "year",
            "AGE": "age",
            "sex_label": "gender",
            "RACE": "race",
        }
    )[["year", "state", "age", "gender", "race", "voter_registration"]]
    return subset.dropna(subset=["year", "state", "age", "gender", "race", "voter_registration"])


def main():
    raw = load_raw_dataframe()
    clean = clean_for_modeling(raw)
    subset = create_individual_response_subset(clean)

    print("\nVOTER REGISTRATION PREVIEW")
    print(clean.head(10).to_string(index=False))
    print("\nREGISTRATION DISTRIBUTION")
    print(subset["voter_registration"].value_counts(dropna=False))

    out_dir = PROJECT_ROOT / "data/processed/voter_reg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "voter_reg_individual_response_subset.csv"
    subset.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
