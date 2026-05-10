"""
Combined CPS voter registration and social-media analysis.

Builds state–year panels, descriptive figures, correlations, K-means clustering
on social features, and a simple linear model for registration rate only.
Turnout is not used.

The registration-by-year figure uses the full voter panel (all CPS years).
Merged tables and scatter plots use an inner join on (state, year), so years
appear only where both voter aggregates and tweets exist.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOTER_PATH = PROJECT_ROOT / "data/processed/voter_reg/voter_reg_individual_response_subset.csv"
BIDEN_PATH = PROJECT_ROOT / "data/processed/social_media/biden_preprocessed.csv"
TRUMP_PATH = PROJECT_ROOT / "data/processed/social_media/trump_preprocessed.csv"
OUTPUT_DATA_DIR = PROJECT_ROOT / "data/processed/analysis"
OUTPUT_FIG_DIR = PROJECT_ROOT / "reports/figures/combined_analysis"


def build_voter_state_year() -> pd.DataFrame:
    """Mean registration rate and row counts per state and CPS survey year."""
    voter = pd.read_csv(VOTER_PATH)
    return (
        voter.groupby(["state", "year"], as_index=False)
        .agg(
            registration_rate=("voter_registration", "mean"),
            n_voter_records=("voter_registration", "size"),
        )
    )


def score_sentiment(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """Attach calendar year and VADER compound sentiment to each tweet row."""
    work = df.copy()
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce")
    work["year"] = work["created_at"].dt.year
    work = work.dropna(subset=["state", "year", "tweet"]).copy()
    work["year"] = work["year"].astype(int)
    work["sentiment"] = work["tweet"].astype(str).map(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    return work


def build_social_state_year() -> pd.DataFrame:
    """Tweet volume and sentiment aggregates per state and calendar year."""
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


def save_descriptive_plots(
    voter_state_year: pd.DataFrame,
    social_state_year: pd.DataFrame,
    merged: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    yearly_voter = voter_state_year.groupby("year", as_index=False)["registration_rate"].mean()

    plt.figure(figsize=(7, 4))
    sns.barplot(data=yearly_voter, x="year", y="registration_rate", color="#4C72B0")
    plt.title("Mean registration rate by year (CPS subset, all states)")
    plt.ylabel("Registration rate")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "registration_rate_by_year.png", dpi=150)
    plt.close()

    yearly_social = social_state_year.groupby("year", as_index=False)["mean_sentiment"].mean()

    plt.figure(figsize=(7, 4))
    sns.barplot(data=yearly_social, x="year", y="mean_sentiment", color="#55A868")
    plt.title("Mean VADER sentiment by year (combined tweet files)")
    plt.ylabel("Mean compound sentiment")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "mean_sentiment_by_year.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.regplot(
        data=merged,
        x="mean_sentiment",
        y="registration_rate",
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "red"},
    )
    plt.title("Mean sentiment vs registration (merged state-year)")
    plt.xlabel("Mean compound sentiment")
    plt.ylabel("Registration rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "mean_sentiment_vs_registration_rate.png", dpi=150)
    plt.close()


def add_social_kmeans_clusters(
    merged: pd.DataFrame, n_clusters: int = 3
) -> pd.DataFrame:
    """
    Cluster state-year rows on standardized tweet_volume and mean_sentiment.
    Rows with missing features get NaN cluster id. k is reduced if sample is small.
    """
    out = merged.copy()
    mask = out[["tweet_volume", "mean_sentiment"]].notna().all(axis=1)
    out["social_cluster"] = pd.NA

    n_valid = int(mask.sum())
    if n_valid < 2:
        return out

    k = min(n_clusters, n_valid)
    features = out.loc[mask, ["tweet_volume", "mean_sentiment"]].values
    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    out.loc[mask, "social_cluster"] = labels.astype(int)
    return out


def save_cluster_plots(labeled: pd.DataFrame) -> None:
    """Cluster size bar chart and mean registration rate by cluster."""
    sns.set_theme(style="whitegrid")
    valid = labeled.dropna(subset=["social_cluster"]).copy()
    if valid.empty:
        return

    counts = valid["social_cluster"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    sns.barplot(x=counts.index.astype(int), y=counts.values, color="#9b59b6")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of state-year rows")
    plt.title("K-means clusters (features: tweet volume, mean sentiment)")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "cluster_sizes.png", dpi=150)
    plt.close()

    by_cluster = (
        valid.groupby("social_cluster", as_index=False)
        .agg(mean_registration=("registration_rate", "mean"))
    )
    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=by_cluster,
        x="social_cluster",
        y="mean_registration",
        color="#34495e",
    )
    plt.xlabel("Cluster ID")
    plt.ylabel("Mean registration rate")
    plt.title("Mean CPS registration rate by social-media cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "registration_rate_by_cluster.png", dpi=150)
    plt.close()


def run_simple_registration_model(merged: pd.DataFrame) -> pd.DataFrame:
    """Exploratory: registration_rate ~ tweet_volume + mean_sentiment."""
    model_df = merged.dropna(subset=["tweet_volume", "mean_sentiment", "registration_rate"]).copy()
    x_features = model_df[["tweet_volume", "mean_sentiment"]]
    y_target = model_df["registration_rate"]
    model = LinearRegression()
    model.fit(x_features, y_target)
    preds = model.predict(x_features)
    rmse = mean_squared_error(y_target, preds) ** 0.5
    return pd.DataFrame(
        {
            "metric": ["r2", "rmse", "coef_tweet_volume", "coef_mean_sentiment", "intercept"],
            "value": [
                r2_score(y_target, preds),
                rmse,
                float(model.coef_[0]),
                float(model.coef_[1]),
                float(model.intercept_),
            ],
        }
    )


def main() -> None:
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    voter_state_year = build_voter_state_year()
    social_state_year = build_social_state_year()
    merged = voter_state_year.merge(social_state_year, on=["state", "year"], how="inner")

    corr_cols = ["registration_rate", "tweet_volume", "mean_sentiment", "sentiment_std"]
    pearson_corr = merged[corr_cols].corr(method="pearson")
    spearman_corr = merged[corr_cols].corr(method="spearman")

    model_results = run_simple_registration_model(merged)
    labeled = add_social_kmeans_clusters(merged, n_clusters=3)
    save_descriptive_plots(voter_state_year, social_state_year, merged)
    save_cluster_plots(labeled)

    voter_state_year.to_csv(OUTPUT_DATA_DIR / "voter_state_year.csv", index=False)
    social_state_year.to_csv(OUTPUT_DATA_DIR / "social_state_year.csv", index=False)
    merged.to_csv(OUTPUT_DATA_DIR / "merged_state_year.csv", index=False)
    labeled.to_csv(OUTPUT_DATA_DIR / "merged_state_year_with_clusters.csv", index=False)
    pearson_corr.to_csv(OUTPUT_DATA_DIR / "correlation_pearson.csv")
    spearman_corr.to_csv(OUTPUT_DATA_DIR / "correlation_spearman.csv")
    model_results.to_csv(OUTPUT_DATA_DIR / "linear_model_registration_results.csv", index=False)

    print("Saved analysis datasets and figures:")
    print(f"- {OUTPUT_DATA_DIR / 'voter_state_year.csv'}")
    print(f"- {OUTPUT_DATA_DIR / 'social_state_year.csv'}")
    print(f"- {OUTPUT_DATA_DIR / 'merged_state_year.csv'}")
    print(f"- {OUTPUT_DATA_DIR / 'merged_state_year_with_clusters.csv'}")
    print(f"- {OUTPUT_DATA_DIR / 'correlation_pearson.csv'}")
    print(f"- {OUTPUT_DATA_DIR / 'correlation_spearman.csv'}")
    print(f"- {OUTPUT_DATA_DIR / 'linear_model_registration_results.csv'}")
    print(f"- {OUTPUT_FIG_DIR / 'registration_rate_by_year.png'}")
    print(f"- {OUTPUT_FIG_DIR / 'mean_sentiment_by_year.png'}")
    print(f"- {OUTPUT_FIG_DIR / 'mean_sentiment_vs_registration_rate.png'}")
    print(f"- {OUTPUT_FIG_DIR / 'cluster_sizes.png'}")
    print(f"- {OUTPUT_FIG_DIR / 'registration_rate_by_cluster.png'}")
    print(f"Merged rows: {len(merged):,}")


if __name__ == "__main__":
    main()
