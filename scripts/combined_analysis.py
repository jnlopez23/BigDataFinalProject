"""Execute combined analysis: merge, visuals, correlations."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOTER_PATH = PROJECT_ROOT / "data/processed/voter_reg/voter_reg_individual_response_subset.csv"
BIDEN_PATH = PROJECT_ROOT / "data/processed/social_media/biden_preprocessed.csv"
TRUMP_PATH = PROJECT_ROOT / "data/processed/social_media/trump_preprocessed.csv"

OUTPUT_DATA_DIR = PROJECT_ROOT / "data/processed/analysis"
OUTPUT_FIG_DIR = PROJECT_ROOT / "reports/figures/combined_analysis"


def build_voter_state_year() -> pd.DataFrame:
    voter = pd.read_csv(VOTER_PATH)
    voter_state_year = (
        voter.groupby(["state", "year"], as_index=False)
        .agg(
            registration_rate=("voter_registration", "mean"),
            turnout_rate=("voting_turnout", "mean"),
            n_voter_records=("voter_registration", "size"),
        )
    )
    return voter_state_year


def score_sentiment(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    work = df.copy()
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce")
    work["year"] = work["created_at"].dt.year
    work = work.dropna(subset=["state", "year", "tweet"]).copy()
    work["year"] = work["year"].astype(int)
    work["sentiment"] = work["tweet"].astype(str).map(lambda t: analyzer.polarity_scores(t)["compound"])
    return work


def build_social_state_year() -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    biden = score_sentiment(pd.read_csv(BIDEN_PATH), analyzer)
    trump = score_sentiment(pd.read_csv(TRUMP_PATH), analyzer)
    social = pd.concat([biden, trump], ignore_index=True)
    social_state_year = (
        social.groupby(["state", "year"], as_index=False)
        .agg(
            tweet_volume=("tweet", "size"),
            mean_sentiment=("sentiment", "mean"),
            sentiment_std=("sentiment", "std"),
        )
    )
    social_state_year["sentiment_std"] = social_state_year["sentiment_std"].fillna(0.0)
    return social_state_year


def save_descriptive_plots(merged: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    yearly = (
        merged.groupby("year", as_index=False)
        .agg(registration_rate=("registration_rate", "mean"), turnout_rate=("turnout_rate", "mean"))
    )

    plt.figure(figsize=(7, 4))
    sns.barplot(data=yearly, x="year", y="registration_rate", color="#4C72B0")
    plt.title("Average Registration Rate by Year")
    plt.ylabel("Registration rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "registration_rate_by_year.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.barplot(data=yearly, x="year", y="turnout_rate", color="#55A868")
    plt.title("Average Turnout Rate by Year")
    plt.ylabel("Turnout rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "turnout_rate_by_year.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.regplot(data=merged, x="mean_sentiment", y="turnout_rate", scatter_kws={"alpha": 0.7}, line_kws={"color": "red"})
    plt.title("Mean Sentiment vs Turnout Rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "mean_sentiment_vs_turnout.png", dpi=150)
    plt.close()


def run_simple_model(merged: pd.DataFrame) -> pd.DataFrame:
    model_df = merged.dropna(subset=["tweet_volume", "mean_sentiment", "turnout_rate"]).copy()
    x = model_df[["tweet_volume", "mean_sentiment"]]
    y = model_df["turnout_rate"]
    model = LinearRegression()
    model.fit(x, y)
    preds = model.predict(x)
    rmse = mean_squared_error(y, preds) ** 0.5
    results = pd.DataFrame(
        {
            "metric": ["r2", "rmse", "coef_tweet_volume", "coef_mean_sentiment", "intercept"],
            "value": [r2_score(y, preds), rmse, model.coef_[0], model.coef_[1], model.intercept_],
        }
    )
    return results


def main() -> None:
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    voter_state_year = build_voter_state_year()
    social_state_year = build_social_state_year()
    merged = voter_state_year.merge(social_state_year, on=["state", "year"], how="inner")

    corr_cols = ["registration_rate", "turnout_rate", "tweet_volume", "mean_sentiment", "sentiment_std"]
    pearson_corr = merged[corr_cols].corr(method="pearson")
    spearman_corr = merged[corr_cols].corr(method="spearman")

    model_results = run_simple_model(merged)
    save_descriptive_plots(merged)

    voter_state_year.to_csv(OUTPUT_DATA_DIR / "voter_state_year.csv", index=False)
    social_state_year.to_csv(OUTPUT_DATA_DIR / "social_state_year.csv", index=False)
    merged.to_csv(OUTPUT_DATA_DIR / "merged_state_year.csv", index=False)
    pearson_corr.to_csv(OUTPUT_DATA_DIR / "correlation_pearson.csv")
    spearman_corr.to_csv(OUTPUT_DATA_DIR / "correlation_spearman.csv")
    model_results.to_csv(OUTPUT_DATA_DIR / "linear_model_turnout_results.csv", index=False)

    print("Saved analysis datasets and figures:")
    print(f"- {OUTPUT_DATA_DIR / 'voter_state_year.csv'}")
    print(f"- {OUTPUT_DATA_DIR / 'social_state_year.csv'}")
    print(f"- {OUTPUT_DATA_DIR / 'merged_state_year.csv'}")
    print(f"- {OUTPUT_DATA_DIR / 'correlation_pearson.csv'}")
    print(f"- {OUTPUT_DATA_DIR / 'correlation_spearman.csv'}")
    print(f"- {OUTPUT_DATA_DIR / 'linear_model_turnout_results.csv'}")
    print(f"- {OUTPUT_FIG_DIR / 'registration_rate_by_year.png'}")
    print(f"- {OUTPUT_FIG_DIR / 'turnout_rate_by_year.png'}")
    print(f"- {OUTPUT_FIG_DIR / 'mean_sentiment_vs_turnout.png'}")
    print(f"Merged rows: {len(merged):,}")


if __name__ == "__main__":
    main()
