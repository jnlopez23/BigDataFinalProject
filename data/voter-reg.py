"""Load and clean CPS voter registration data for ML analysis."""

# imports
from pathlib import Path
import pandas as pd

# Fixed-width layout for abbreviated data set with data from Nov 2020, 2022, and 2024 (abrv_voter_reg.dat)
SPECS = {
    "YEAR": (1, 4),
    "STATEFIP": (50, 51),
    "METFIPS": (59, 63),
    "FAMINC": (70, 72),
    "AGE": (129, 130),
    "SEX": (131, 131),
    "RACE": (132, 134),
    "VOTED": (155, 156),
    "VOREG": (157, 158),
    "VOYNOTREG": (159, 160),
    "VOWHYNOT": (161, 162),
    "VOTEHOW": (163, 164),
    "VOREGHOW": (165, 166),
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
VOTEHOW_LABELS = {1: "In person", 2: "By mail"}
VOREGHOW_LABELS = {
    1: "DMV",
    2: "Public assistance agency",
    3: "Mail",
    4: "School/hospital/campus",
    5: "Gov registration office",
    6: "Registration drive",
    7: "Polling place",
    8: "Internet",
    9: "Other",
}


def _build_colspecs(specs: dict[str, tuple[int, int]]) -> tuple[list[tuple[int, int]], list[str]]:
    names = list(specs.keys())
    colspecs = [(start - 1, end) for start, end in specs.values()]
    return colspecs, names


def _resolve_data_file() -> Path:
    candidates = [
        Path("data/files/abrv_voter_reg.dat"),
        Path("data/files/cps_00002.dat"),
        Path("data/voter_reg.dat"),
        Path("voter_reg.dat"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate abrv_voter_reg.dat (or fallback data files) in expected paths."
    )


def load_raw_dataframe() -> pd.DataFrame:
    colspecs, names = _build_colspecs(SPECS)
    data_path = _resolve_data_file()
    return pd.read_fwf(data_path, colspecs=colspecs, names=names, dtype=str)


def clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "YEAR",
        "STATEFIP",
        "METFIPS",
        "FAMINC",
        "AGE",
        "SEX",
        "RACE",
        "VOTED",
        "VOREG",
        "VOYNOTREG",
        "VOWHYNOT",
        "VOTEHOW",
        "VOREGHOW",
        "WTFINL",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Adjust for invalid responses (mark as missing))
    missing_by_col = {
        "VOTED": [96, 97, 98, 99],
        "VOREG": [96, 97, 98, 99],
        "VOYNOTREG": [96, 97, 98, 99],
        "VOWHYNOT": [96, 97, 98, 99],
        "VOTEHOW": [96, 97, 98, 99],
        "VOREGHOW": [96, 97, 98, 99],
        "FAMINC": [995, 996, 997, 999],
    }
    for col, codes in missing_by_col.items():
        df.loc[df[col].isin(codes), col] = pd.NA

    # Keep voting-age civilian records for turnout analysis.
    df = df.loc[df["AGE"] >= 18].copy()

    # Core modeling targets/features.
    df["voted_binary"] = df["VOTED"].map({1: 0, 2: 1})
    df["registered_binary"] = df["VOREG"].map({1: 0, 2: 1})
    df["state"] = df["STATEFIP"].map(STATEFIP_LABELS)
    df["sex_label"] = df["SEX"].map(SEX_LABELS)
    df["vote_method"] = df["VOTEHOW"].map(VOTEHOW_LABELS)
    df["registration_method"] = df["VOREGHOW"].map(VOREGHOW_LABELS)

    # Common numeric proxy for bracketed family income code.
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
        "voted_binary",
        "registered_binary",
        "vote_method",
        "registration_method",
        "VOYNOTREG",
        "VOWHYNOT",
        "WTFINL",
    ]
    return df[keep_cols]


def create_individual_response_subset(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only requested columns with consistent naming style.
    subset = df.rename(
        columns={
            "YEAR": "year",
            "AGE": "age",
            "sex_label": "gender",
            "RACE": "race",
            "registered_binary": "voter_registration",
            "voted_binary": "voting_turnout",
        }
    )[
        [
            "year",
            "state",
            "age",
            "gender",
            "race",
            "voter_registration",
            "voting_turnout",
            "vote_method",
        ]
    ].copy()
    # Clean NaNs for key subset fields.
    return subset.dropna(
        subset=["year", "state", "age", "gender", "race", "voter_registration", "voting_turnout"]
    )


def main() -> None:
    raw = load_raw_dataframe()
    clean = clean_for_modeling(raw)
    subset = create_individual_response_subset(clean)

    print("\nDATA PREVIEW")
    print(clean.head(10).to_string(index=False))
    print("\nCOLUMN DTYPES")
    print(clean.dtypes.to_string())
    print("\nRESPONSE PREVIEW")
    print("voting_turnout counts:")
    print(subset["voting_turnout"].value_counts(dropna=False).sort_index().to_string())
    print("\nvoter_registration counts:")
    print(subset["voter_registration"].value_counts(dropna=False).sort_index().to_string())
    print("\nvote_method counts:")
    print(subset["vote_method"].value_counts(dropna=False).to_string())
    print("\nRows in requested subset:")
    print(f"{len(subset):,}")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_dir / "voter_reg_individual_response_subset.csv", index=False)

    print("Saved:")
    print(f"- {output_dir / 'voter_reg_individual_response_subset.csv'}")


if __name__ == "__main__":
    main()