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
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
CLUSTER_FEATURES = ("tweet_volume", "mean_sentiment")
K_PCA_SWEEP_MAX = 5


def _padded_xlim_years(years) -> tuple[float, float]:
    """Widen x-axis so end years are not flush against the plot edge."""
    arr = np.unique(np.asarray(years, dtype=float))
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0, 1.0
    lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo if hi > lo else 1.0
    pad = max(span * 0.14, 0.65)
    return lo - pad, hi + pad


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

    yearly_voter = (
        voter_state_year.groupby("year", as_index=False)["registration_rate"]
        .mean()
        .sort_values("year")
    )
    yearly_merged_reg = (
        merged.groupby("year", as_index=False)["registration_rate"]
        .mean()
        .sort_values("year")
    )

    plt.figure(figsize=(8.5, 4))
    plt.plot(
        yearly_voter["year"],
        yearly_voter["registration_rate"],
        color="#4C72B0",
        linewidth=2,
        marker="o",
        markersize=9,
        markeredgecolor="white",
        markeredgewidth=1,
        label="All CPS state-years",
    )
    if not yearly_merged_reg.empty:
        plt.plot(
            yearly_merged_reg["year"],
            yearly_merged_reg["registration_rate"],
            color="#e74c3c",
            linewidth=2,
            marker="s",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1,
            label="Merged with tweets (inner join)",
        )
    reg_years = (
        pd.concat([yearly_voter["year"], yearly_merged_reg["year"]], ignore_index=True)
        if not yearly_merged_reg.empty
        else yearly_voter["year"]
    )
    x_lo, x_hi = _padded_xlim_years(reg_years)
    plt.xlim(x_lo, x_hi)
    year_ticks = sorted(pd.unique(reg_years.astype(int)))
    plt.xticks(year_ticks)
    plt.title("Mean Registration Rate by Year")
    plt.ylabel("Registration Rate")
    plt.xlabel("Year")
    plt.legend(loc="best", framealpha=0.95)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "registration_rate_by_year.png", dpi=150)
    plt.close()

    yearly_social = (
        social_state_year.groupby("year", as_index=False)["mean_sentiment"]
        .mean()
        .sort_values("year")
    )

    plt.figure(figsize=(8.5, 4))
    sy = yearly_social["year"].astype(int).to_numpy()
    sm = yearly_social["mean_sentiment"].to_numpy(dtype=float)
    plt.bar(
        sy,
        sm,
        width=0.65,
        color="#55A868",
        edgecolor="white",
        linewidth=1,
    )
    sx_lo, sx_hi = _padded_xlim_years(yearly_social["year"])
    plt.xlim(sx_lo, sx_hi)
    plt.xticks(sorted(yearly_social["year"].astype(int).unique()))
    plt.title("Mean VADER Sentiment by Year (Combined Tweet Files)")
    plt.ylabel("Mean Compound Sentiment")
    plt.xlabel("Year")
    plt.grid(True, linestyle=":", alpha=0.6)
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
    plt.title("Mean Sentiment vs Registration (Merged State-year)")
    plt.xlabel("Mean Compound Sentiment")
    plt.ylabel("Registration Rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "mean_sentiment_vs_registration_rate.png", dpi=150)
    plt.close()


def add_social_kmeans_clusters(
    merged: pd.DataFrame, n_clusters: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Cluster state-year rows on standardized tweet_volume and mean_sentiment.
    Rows with missing features get NaN cluster id. k is reduced if sample is small.
    Adds PC1, PC2 from PCA on the same scaled features for visualization.
    """
    out = merged.copy()
    mask = out[list(CLUSTER_FEATURES)].notna().all(axis=1)
    out["social_cluster"] = pd.NA
    out["PC1"] = np.nan
    out["PC2"] = np.nan

    n_valid = int(mask.sum())
    if n_valid < 2:
        return out, None

    k = min(n_clusters, n_valid)
    features = out.loc[mask, list(CLUSTER_FEATURES)].values
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2)
    pca_xy = pca.fit_transform(scaled)
    out.loc[mask, "PC1"] = pca_xy[:, 0]
    out.loc[mask, "PC2"] = pca_xy[:, 1]
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    out.loc[mask, "social_cluster"] = labels.astype(int)
    loadings = pd.DataFrame(
        pca.components_,
        columns=list(CLUSTER_FEATURES),
        index=["PC1", "PC2"],
    )
    return out, loadings


def save_cluster_plots(labeled: pd.DataFrame) -> None:
    """Cluster sizes and mean registration by cluster: line+markers if 2+ clusters, else bar."""
    sns.set_theme(style="whitegrid")
    valid = labeled.dropna(subset=["social_cluster"]).copy()
    if valid.empty:
        return

    counts = valid["social_cluster"].value_counts().sort_index()
    c_ids = counts.index.astype(int).to_numpy()
    c_vals = counts.values.astype(float)

    plt.figure(figsize=(7, 4))
    if len(c_ids) == 1:
        plt.bar([int(c_ids[0])], [c_vals[0]], width=0.45, color="#9b59b6", edgecolor="white", linewidth=1)
        plt.xticks([int(c_ids[0])])
    else:
        plt.plot(
            c_ids,
            c_vals,
            color="#9b59b6",
            linewidth=2,
            marker="o",
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1,
        )
        plt.xticks(c_ids)
        x_lo, x_hi = float(c_ids.min()), float(c_ids.max())
        span = max(x_hi - x_lo, 1.0)
        plt.margins(x=max(0.08 * span, 0.25))
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of state-year rows")
    plt.title("K-means clusters (features: tweet volume, mean sentiment)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "cluster_sizes.png", dpi=150)
    plt.close()

    by_cluster = (
        valid.groupby("social_cluster", as_index=False)
        .agg(mean_registration=("registration_rate", "mean"))
        .sort_values("social_cluster")
    )
    bc_ids = by_cluster["social_cluster"].astype(int).to_numpy()
    bc_vals = by_cluster["mean_registration"].to_numpy(dtype=float)

    plt.figure(figsize=(7, 4))
    if len(bc_ids) == 1:
        plt.bar(
            [int(bc_ids[0])],
            [bc_vals[0]],
            width=0.45,
            color="#34495e",
            edgecolor="white",
            linewidth=1,
        )
        plt.xticks([int(bc_ids[0])])
    else:
        plt.plot(
            bc_ids,
            bc_vals,
            color="#34495e",
            linewidth=2,
            marker="o",
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1,
        )
        plt.xticks(bc_ids)
        x_lo, x_hi = float(bc_ids.min()), float(bc_ids.max())
        span = max(x_hi - x_lo, 1.0)
        plt.margins(x=max(0.08 * span, 0.25))
    plt.xlabel("Cluster ID")
    plt.ylabel("Mean Registration Rate")
    plt.title("Mean CPS Registration Rate by Social Media Cluster")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "registration_rate_by_cluster.png", dpi=150)
    plt.close()


def save_cluster_pca_plots(labeled: pd.DataFrame, k_sweep_max: int = K_PCA_SWEEP_MAX) -> None:
    """PCA scatter of social features, colored by pipeline clusters; sweep k up to k_sweep_max."""
    sns.set_theme(style="whitegrid")
    mask = labeled[list(CLUSTER_FEATURES)].notna().all(axis=1) & labeled["social_cluster"].notna()
    sub = labeled.loc[mask]
    if sub.empty:
        return

    pc1 = sub["PC1"].to_numpy(dtype=float)
    pc2 = sub["PC2"].to_numpy(dtype=float)
    cluster_ids = sub["social_cluster"].astype(int).to_numpy()
    n = len(sub)

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(pc1, pc2, c=cluster_ids, cmap="plasma", alpha=0.6, edgecolors="none")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("K-Means clusters visualized by PCA (pipeline k)")
    plt.colorbar(sc, label="Cluster")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "cluster_pca_pipeline.png", dpi=150)
    plt.close()

    X = sub[list(CLUSTER_FEATURES)].values
    scaled = StandardScaler().fit_transform(X)
    k_hi = min(k_sweep_max, n)
    for k in range(1, k_hi + 1):
        plt.figure(figsize=(6, 5))
        labels_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        scatter = plt.scatter(
            pc1,
            pc2,
            c=labels_k,
            cmap="tab10",
            alpha=0.7,
            edgecolors="none",
        )
        plt.title(f"PCA visualization of K-Means clusters (k = {k})", fontsize=14)
        plt.xlabel("Principal Component 1 (PC1)", fontsize=12)
        plt.ylabel("Principal Component 2 (PC2)", fontsize=12)
        plt.colorbar(scatter, label="Cluster label")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG_DIR / f"cluster_pca_k{k}.png", dpi=150)
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
    labeled, pca_loadings = add_social_kmeans_clusters(merged, n_clusters=3)
    save_descriptive_plots(voter_state_year, social_state_year, merged)
    save_cluster_plots(labeled)
    save_cluster_pca_plots(labeled)

    voter_state_year.to_csv(OUTPUT_DATA_DIR / "voter_state_year.csv", index=False)
    social_state_year.to_csv(OUTPUT_DATA_DIR / "social_state_year.csv", index=False)
    merged.to_csv(OUTPUT_DATA_DIR / "merged_state_year.csv", index=False)
    labeled.to_csv(OUTPUT_DATA_DIR / "merged_state_year_with_clusters.csv", index=False)
    pearson_corr.to_csv(OUTPUT_DATA_DIR / "correlation_pearson.csv")
    spearman_corr.to_csv(OUTPUT_DATA_DIR / "correlation_spearman.csv")
    model_results.to_csv(OUTPUT_DATA_DIR / "linear_model_registration_results.csv", index=False)
    if pca_loadings is not None:
        pca_loadings.to_csv(OUTPUT_DATA_DIR / "cluster_pca_loadings.csv")

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
    print(f"- {OUTPUT_FIG_DIR / 'cluster_pca_pipeline.png'}")
    print(
        f"- {OUTPUT_FIG_DIR / 'cluster_pca_k1.png'} … "
        f"{OUTPUT_FIG_DIR / f'cluster_pca_k{K_PCA_SWEEP_MAX}.png'} (k sweep 1–{K_PCA_SWEEP_MAX})"
    )
    print(f"Merged rows: {len(merged):,}")


if __name__ == "__main__":
    main()
