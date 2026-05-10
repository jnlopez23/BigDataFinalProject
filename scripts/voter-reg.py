"""Load and clean CPS voter registration data for ML analysis."""

import pandas as pd
from pathlib import Path


def _build_colspecs(specs):
    """
    Convert fixed-width column spec dictionary into colspecs format for pandas.
    """
    names = list(specs.keys())
    colspecs = [(s - 1, e) for s, e in specs.values()]
    return colspecs, names


def _resolve_data_file():
    """
    Locate CPS voter registration dataset from expected project directories.
    """
    candidates = [
        Path("data/files/abrv_voter_reg.dat"),
        Path("data/voter_reg.dat"),
        Path("abrv_voter_reg.dat"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not locate voter registration dataset.")


def load_raw_dataframe():
    """
    Load raw CPS voter registration dataset using fixed-width format parsing.
    """
    colspecs, names = _build_colspecs(SPECS)
    data_path = _resolve_data_file()
    return pd.read_fwf(data_path, colspecs=colspecs, names=names, dtype=str)


def clean_for_modeling(df):
    """
    Clean CPS voter registration data for modeling.

    - Convert selected columns to numeric
    - Remove invalid response codes
    - Filter to adults (18+)
    - Create binary voter registration variable
    - Map demographic labels
    """
    numeric_cols = [
        "YEAR", "STATEFIP", "METFIPS", "FAMINC",
        "AGE", "SEX", "RACE", "VOREG", "WTFINL"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # remove invalid/missing response codes
    df.loc[df["VOREG"].isin([96, 97, 98, 99]), "VOREG"] = pd.NA
    df.loc[df["FAMINC"].isin([995, 996, 997, 999]), "FAMINC"] = pd.NA

    # keep adults only
    df = df.loc[df["AGE"] >= 18].copy()

    # create target variable
    df["voter_registration"] = df["VOREG"].map({1: 0, 2: 1})

    # map labels
    df["state"] = df["STATEFIP"].map(STATEFIP_LABELS)
    df["sex_label"] = df["SEX"].map(SEX_LABELS)

    # keep income code as-is
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
    """
    Create final analysis-ready subset for modeling and visualization.
    """
    subset = df.rename(
        columns={
            "YEAR": "year",
            "AGE": "age",
            "sex_label": "gender",
            "RACE": "race",
        }
    )[["year", "state", "age", "gender", "race", "voter_registration"]]

    return subset.dropna(
        subset=["year", "state", "age", "gender", "race", "voter_registration"]
    )


def main():
    """
    Run full preprocessing pipeline:
    - Load raw data
    - Clean dataset
    - Create modeling subset
    - Save processed output
    """
    raw = load_raw_dataframe()
    clean = clean_for_modeling(raw)
    subset = create_individual_response_subset(clean)

    print("\nVOTER REGISTRATION PREVIEW")
    print(clean.head(10).to_string(index=False))

    print("\nREGISTRATION DISTRIBUTION")
    print(subset["voter_registration"].value_counts(dropna=False))

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    subset.to_csv(output_dir / "voter_registration_clean.csv", index=False)

    print("\nSaved to data/processed/voter_registration_clean.csv")


if __name__ == "__main__":
    main()