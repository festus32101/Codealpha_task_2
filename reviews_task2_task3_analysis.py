#!/usr/bin/env python3

from __future__ import annotations
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

EXPECTED_COLUMNS = [
    "Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator",
    "HelpfulnessDenominator", "Score", "Time", "Summary", "Text"
]

def safe_write_csv(frame, path):
    frame.to_csv(path, index=True, index_label="Variable")

def main():
    parser = argparse.ArgumentParser(description="Explore Amazon fine food reviews.")
    parser.add_argument("--input", default="Reviews.csv", help="Input CSV path (default: Reviews.csv)")
    parser.add_argument("--output-dir", default="reviews_analysis_output", help="Folder for outputs")
    parser.add_argument("--sample-size", type=int, default=100000,
                        help="Maximum rows used for text-length and selected plots; 0 uses all rows")
    args = parser.parse_args()

    source = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(
            f"Cannot find {source}. Put this script beside Reviews.csv or use --input /path/to/Reviews.csv"
        )

    print(f"Reading {source} ...")
    df = pd.read_csv(source, low_memory=False)
    report = []
    report.append("AMAZON FINE FOOD REVIEWS — EXPLORATORY ANALYSIS")
    report.append("=" * 58)
    report.append(f"Source file: {source.resolve()}")
    report.append(f"Rows: {len(df):,}")
    report.append(f"Columns: {df.shape[1]}")
    report.append("")

    # 1. Data structure and variable types
    structure = pd.DataFrame({
        "data_type": df.dtypes.astype(str),
        "non_null_count": df.notna().sum(),
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(4),
        "unique_values": df.nunique(dropna=True),
    })
    safe_write_csv(structure, out / "01_variable_structure.csv")
    report.append("1. DATA STRUCTURE")
    report.append(structure.to_string())
    report.append("")

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(4),
    }).sort_values("missing_count", ascending=False)
    safe_write_csv(missing, out / "02_missing_values.csv")

    # 2. Data quality checks. These are flagged, not silently repaired.
    duplicate_id_count = int(df["Id"].duplicated().sum()) if "Id" in df else None
    exact_duplicate_rows = int(df.duplicated().sum())
    duplicate_review_key = None
    review_key_cols = [c for c in ["ProductId", "UserId", "Time", "Text"] if c in df.columns]
    if len(review_key_cols) == 4:
        duplicate_review_key = int(df.duplicated(subset=review_key_cols).sum())

    invalid_score = int((~df["Score"].between(1, 5)).sum()) if "Score" in df else None
    invalid_help = int((df["HelpfulnessNumerator"] > df["HelpfulnessDenominator"]).sum())
    negative_help = int(((df["HelpfulnessNumerator"] < 0) | (df["HelpfulnessDenominator"] < 0)).sum())
    zero_denominator = int((df["HelpfulnessDenominator"] == 0).sum())
    blank_profile = int(df["ProfileName"].astype("string").str.strip().eq("").fillna(False).sum()) if "ProfileName" in df else 0
    blank_summary = int(df["Summary"].astype("string").str.strip().eq("").fillna(False).sum()) if "Summary" in df else 0
    blank_text = int(df["Text"].astype("string").str.strip().eq("").fillna(False).sum()) if "Text" in df else 0

    data_issues = pd.DataFrame([
        ["Duplicate Id values", duplicate_id_count, "Id should normally identify one row; investigate before removing anything."],
        ["Exact duplicate rows", exact_duplicate_rows, "Rows identical across every column."],
        ["Repeated review key (ProductId, UserId, Time, Text)", duplicate_review_key, "Potential repeat submissions; not automatically removed."],
        ["Scores outside 1–5", invalid_score, "Ratings should be within the documented 1–5 scale."],
        ["Helpfulness numerator > denominator", invalid_help, "Logically inconsistent helpfulness counts; verify source."],
        ["Negative helpfulness counts", negative_help, "Counts should not be negative."],
        ["Zero helpfulness denominator", zero_denominator, "Usually means no votes were recorded; ratio is undefined, not zero."],
        ["Blank ProfileName strings", blank_profile, "Different from missing/NA values."],
        ["Blank Summary strings", blank_summary, "Different from missing/NA values."],
        ["Blank Text strings", blank_text, "Review text may be required for text analysis."],
    ], columns=["check", "count", "interpretation"])
    data_issues.to_csv(out / "03_data_quality_flags.csv", index=False)

    report.append("2. DATA QUALITY FLAGS (flags are not automatic corrections)")
    report.append(data_issues.to_string(index=False))
    report.append("")

    # 3. Derived variables, preserving the original timestamp
    work = df.copy()
    work["ReviewDate"] = pd.to_datetime(work["Time"], unit="s", errors="coerce", utc=True).dt.date
    work["ReviewYear"] = pd.to_datetime(work["Time"], unit="s", errors="coerce", utc=True).dt.year
work["ReviewMonth"] = (
    pd.to_datetime(work["Time"], unit="s", errors="coerce", utc=True)
    .dt.tz_localize(None)
    .dt.to_period("M")
    .astype(str)
)    work["ReviewTextLength"] = work["Text"].fillna("").astype(str).str.len()
    work["SummaryLength"] = work["Summary"].fillna("").astype(str).str.len()
    denom_positive = work["HelpfulnessDenominator"] > 0
    work["HelpfulnessRatio"] = np.where(
        denom_positive,
        work["HelpfulnessNumerator"] / work["HelpfulnessDenominator"],
        np.nan
    )
    work["HasHelpfulnessVotes"] = work["HelpfulnessDenominator"] > 0

    # 4. Descriptive statistics and patterns
    score_summary = work["Score"].describe().to_frame("value")
    score_summary.to_csv(out / "04_score_descriptive_statistics.csv")
    score_counts = work["Score"].value_counts(dropna=False).sort_index().rename_axis("Score").reset_index(name="ReviewCount")
    score_counts["Percent"] = (score_counts["ReviewCount"] / len(work) * 100).round(3)
    score_counts.to_csv(out / "05_rating_distribution.csv", index=False)

    product_summary = work.groupby("ProductId", dropna=False).agg(
        review_count=("Id", "size"),
        average_score=("Score", "mean"),
        median_score=("Score", "median"),
        score_std=("Score", "std")
    ).sort_values("review_count", ascending=False)
    product_summary.head(100).to_csv(out / "06_top_100_products_by_review_count.csv")

    user_counts = work["UserId"].value_counts(dropna=False)
    repeated_users = user_counts[user_counts > 1].rename_axis("UserId").reset_index(name="ReviewCount")
    repeated_users.head(100).to_csv(out / "07_top_100_users_by_review_count.csv", index=False)

    yearly = work.groupby("ReviewYear", dropna=False).agg(
        review_count=("Id", "size"),
        average_score=("Score", "mean"),
        five_star_share=("Score", lambda s: (s == 5).mean() * 100)
    ).reset_index()
    yearly.to_csv(out / "08_yearly_trends.csv", index=False)

    monthly = work.dropna(subset=["ReviewYear"]).groupby("ReviewMonth").agg(
        review_count=("Id", "size"),
        average_score=("Score", "mean")
    ).reset_index()
    monthly.to_csv(out / "09_monthly_trends.csv", index=False)

    # Sample large text columns for memory-efficient plot and text length summaries.
    if args.sample_size and len(work) > args.sample_size:
        sample = work.sample(args.sample_size, random_state=42)
    else:
        sample = work

    text_stats = sample[["ReviewTextLength", "SummaryLength"]].describe().T
    text_stats.to_csv(out / "10_text_length_statistics.csv")

    # 5. Hypothesis tests with explicit nulls and caveats.
    test_rows = []
    # H1: mean numeric rating equals neutral midpoint 3.0 (descriptive benchmark; rating is ordinal)
    scores = work["Score"].dropna().astype(float)
    if len(scores) > 1:
        t_stat, p_val = stats.ttest_1samp(scores, popmean=3.0)
        sd = scores.std(ddof=1)
        effect_d = (scores.mean() - 3.0) / sd if sd else np.nan
        test_rows.append({
            "hypothesis": "H1: Mean numeric score equals 3.0",
            "test": "One-sample t-test (rating treated approximately numeric)",
            "n": len(scores), "statistic": t_stat, "p_value": p_val,
            "effect_size": effect_d, "estimate": scores.mean(),
            "interpretation": "Compare p-value with 0.05; also assess effect size and ordinal-scale limitation."
        })

    # H2: scores differ between reviews with at least one helpfulness vote and those with none.
    with_votes = work.loc[work["HelpfulnessDenominator"] > 0, "Score"].dropna()
    without_votes = work.loc[work["HelpfulnessDenominator"] == 0, "Score"].dropna()
    if len(with_votes) and len(without_votes):
        u_stat, p_val = stats.mannwhitneyu(with_votes, without_votes, alternative="two-sided")
        # Rank-biserial effect size; sign depends on group ordering.
        effect_rb = 1 - (2 * u_stat) / (len(with_votes) * len(without_votes))
        test_rows.append({
            "hypothesis": "H2: Score distributions are equal for reviews with votes vs no votes",
            "test": "Two-sided Mann–Whitney U test",
            "n": len(with_votes) + len(without_votes), "statistic": u_stat, "p_value": p_val,
            "effect_size": effect_rb, "estimate": with_votes.median() - without_votes.median(),
            "interpretation": "Estimate is median(score with votes) minus median(score without votes); association is not causation."
        })

    # H3: Review text length differs across 1-star and 5-star reviews (sample for manageable testing).
    one_star = sample.loc[sample["Score"] == 1, "ReviewTextLength"]
    five_star = sample.loc[sample["Score"] == 5, "ReviewTextLength"]
    if len(one_star) and len(five_star):
        u_stat, p_val = stats.mannwhitneyu(one_star, five_star, alternative="two-sided")
        test_rows.append({
            "hypothesis": "H3: Review text length distributions are equal for 1-star vs 5-star reviews",
            "test": "Two-sided Mann–Whitney U test on sampled rows",
            "n": len(one_star) + len(five_star), "statistic": u_stat, "p_value": p_val,
            "effect_size": np.nan, "estimate": one_star.median() - five_star.median(),
            "interpretation": "Estimate is median characters (1-star minus 5-star); sample is random_state=42 if dataset is sampled."
        })

    tests = pd.DataFrame(test_rows)
    tests.to_csv(out / "11_hypothesis_tests.csv", index=False)

    # 6. Visualizations
    plt.figure(figsize=(8, 5))
    plt.bar(score_counts["Score"].astype(str), score_counts["ReviewCount"])
    plt.title("Distribution of Review Ratings")
    plt.xlabel("Rating (Score)")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(out / "chart_01_rating_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    yearly_plot = yearly.dropna(subset=["ReviewYear"])
    plt.plot(yearly_plot["ReviewYear"], yearly_plot["review_count"], marker="o")
    plt.title("Review Volume by Year")
    plt.xlabel("Review year (UTC)")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(out / "chart_02_review_volume_by_year.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(sample["ReviewTextLength"].clip(upper=2000), bins=50)
    plt.title("Review Text Length (capped at 2,000 characters)")
    plt.xlabel("Characters in review text")
    plt.ylabel("Frequency in analysis sample")
    plt.tight_layout()
    plt.savefig(out / "chart_03_review_text_length.png", dpi=160)
    plt.close()

    top_products = product_summary.head(10).sort_values("review_count")
    plt.figure(figsize=(10, 6))
    plt.barh(top_products.index.astype(str), top_products["review_count"])
    plt.title("Top 10 Products by Number of Reviews")
    plt.xlabel("Number of reviews")
    plt.ylabel("ProductId")
    plt.tight_layout()
    plt.savefig(out / "chart_04_top_products.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.boxplot([one_star, five_star], tick_labels=["1-star", "5-star"], showfliers=False)
    plt.title("Review Text Length: 1-star vs 5-star")
    plt.ylabel("Characters")
    plt.tight_layout()
    plt.savefig(out / "chart_05_text_length_by_rating.png", dpi=160)
    plt.close()

    # 7. Human-readable report with results and caveats
    report.append("3. DESCRIPTIVE FINDINGS")
    report.append(f"Mean score: {work['Score'].mean():.4f}")
    report.append(f"Median score: {work['Score'].median():.2f}")
    report.append(f"5-star share: {(work['Score'].eq(5).mean() * 100):.2f}%")
    report.append(f"Unique products: {work['ProductId'].nunique(dropna=True):,}")
    report.append(f"Unique users: {work['UserId'].nunique(dropna=True):,}")
    report.append(f"First review timestamp (UTC): {pd.to_datetime(work['Time'], unit='s', errors='coerce', utc=True).min()}")
    report.append(f"Last review timestamp (UTC): {pd.to_datetime(work['Time'], unit='s', errors='coerce', utc=True).max()}")
    report.append(f"Median review text length in analysis sample: {sample['ReviewTextLength'].median():.0f} characters")
    report.append("")
    report.append("4. HYPOTHESIS TESTS")
    if tests.empty:
        report.append("No hypothesis tests could be calculated; check required columns.")
    else:
        report.append(tests.to_string(index=False))
    report.append("")
    report.append("5. INTERPRETATION AND LIMITATIONS")
    report.append("- A very small p-value is not necessarily a practically important effect, especially with a large dataset.")
    report.append("- The 1–5 score is ordinal; a t-test on the mean treats rating steps as approximately equally spaced.")
    report.append("- Statistical significance does not establish causation.")
    report.append("- Review volume reflects this dataset's collection process and should not automatically be treated as all Amazon reviews.")
    report.append("- Timestamps are converted to UTC; local dates may differ.")
    report.append("- Repeated review keys and unusual values are flagged for investigation rather than automatically deleted.")
    report.append("- Text length does not measure sentiment; sentiment analysis would be a separate analysis.")
    (out / "analysis_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("\\nAnalysis complete.")
    print(f"Outputs saved to: {out.resolve()}")
    print("Key files: analysis_report.txt, 01_variable_structure.csv, 03_data_quality_flags.csv,")
    print("           11_hypothesis_tests.csv, and chart_*.png")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
