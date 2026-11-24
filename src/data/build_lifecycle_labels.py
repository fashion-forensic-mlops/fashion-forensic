from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PANEL_PATH = (
    PROJECT_ROOT / "data/processed/weekly_panel.parquet"
)

BOUNDARIES_PATH = (
    PROJECT_ROOT / "data/interim/observation_boundaries.parquet"
)

#Lifecycle label output
OUTPUT_PATH = (
    PROJECT_ROOT / "data/processed/article_lifecycle_labels.parquet"
)

DEFAULT_W = 4
SENSITIVITY_W = 8

EXPECTED_ARTICLES = 104_547
EXPECTED_DEATHS_W4 = 75_658
EXPECTED_CENSORED_W4 = 28_889
EXPECTED_DEATHS_W8 = 68_533
EXPECTED_CENSORED_W8 = 36_014


def validate_inputs():
    for path in [PANEL_PATH, BOUNDARIES_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )


def main():
    print("=" * 60)
    print("M2 - LIFECYCLE LABEL GENERATION")
    print("=" * 60)

    validate_inputs()

    #Load weekly panel
    print("\nLoading weekly panel...")

    panel = pl.scan_parquet(PANEL_PATH)

    #Load observation boundaries
    print("\nLoading observation boundaries...")

    boundaries = pl.read_parquet(BOUNDARIES_PATH)

    LAST_TRANSACTION_DATE = boundaries[
        "last_transaction_date"
    ][0]

    CUTOFF_WEEK = boundaries["cutoff_week"][0]

    LAST_COMPLETE_WEEK = boundaries[
        "last_complete_week"
    ][0]

    print("Last transaction:", LAST_TRANSACTION_DATE)
    print("Final calendar week:", CUTOFF_WEEK)
    print("Last complete week:", LAST_COMPLETE_WEEK)

    #Summarize article sales histories
    print("\nSummarizing article sales histories...")

    article_summary = (
        panel
        .group_by("article_id")
        .agg([
            pl.col("birth_date").first().alias("birth_date"),
            pl.col("birth_week").first().alias("birth_week"),
            pl.col("week_start").min().alias("first_observed_week"),
            pl.col("week_start").max().alias("last_observed_week"),
            pl.col("age_week").max().alias("observed_lifetime_weeks"),
            pl.len().alias("total_observed_weeks"),
            pl.col("units_sold").sum().alias("total_units_sold"),
            (pl.col("units_sold") > 0).sum().alias("weeks_with_sales"),
            (pl.col("units_sold") == 0).sum().alias("zero_sales_weeks"),
            pl.col("week_start")
            .filter(pl.col("units_sold") > 0)
            .max()
            .alias("last_positive_sales_week"),
            pl.col("units_sold").max().alias("maximum_weekly_sales"),
            pl.col("units_sold").mean().alias("average_weekly_sales"),
        ])
        .sort("article_id")
    )

    #Calculate complete zero-sales weeks
    print("\nCalculating complete zero-sales weeks...")

    lifecycle_base = (
        article_summary
        .with_columns(
            pl.when(
                pl.col("last_positive_sales_week")
                <= pl.lit(LAST_COMPLETE_WEEK)
            )
            .then(
                (
                    (
                        pl.lit(LAST_COMPLETE_WEEK)
                        - pl.col("last_positive_sales_week")
                    ).dt.total_days() // 7
                ).cast(pl.Int32)
            )
            .otherwise(pl.lit(0))
            .alias("complete_zero_weeks_after_last_sale")
        )
    )

    #Generate default lifecycle labels (w = 4)
    print(f"\nGenerating default lifecycle labels (w={DEFAULT_W})...")

    lifecycle_w4 = (
        lifecycle_base
        .with_columns([
            (
                pl.col("complete_zero_weeks_after_last_sale")
                >= DEFAULT_W
            ).alias("event_observed_w4"),

            pl.when(
                pl.col("complete_zero_weeks_after_last_sale")
                >= DEFAULT_W
            )
            .then(
                pl.col("last_positive_sales_week")
                + pl.duration(days=7)
            )
            .otherwise(None)
            .alias("death_week_w4"),

            pl.lit(LAST_TRANSACTION_DATE).alias("censor_date"),
        ])
        .with_columns([
            pl.when(pl.col("event_observed_w4"))
            .then(pl.lit("death_observed"))
            .otherwise(pl.lit("right_censored"))
            .alias("lifecycle_status_w4"),

            pl.when(pl.col("event_observed_w4"))
            .then(
                (
                    (
                        pl.col("last_positive_sales_week")
                        + pl.duration(days=7)
                        - pl.col("birth_week")
                    ).dt.total_days() // 7
                ).cast(pl.Int32)
            )
            .otherwise(None)
            .alias("death_age_week_w4"),

            (
                (
                    pl.lit(CUTOFF_WEEK)
                    - pl.col("birth_week")
                ).dt.total_days() // 7
            )
            .cast(pl.Int32)
            .alias("censor_age_week"),
        ])
    )

    #Generate sensitivity lifecycle labels (w = 8)
    print(
        f"\nGenerating sensitivity lifecycle labels "
        f"(w={SENSITIVITY_W})..."
    )

    lifecycle_labels = (
        lifecycle_w4
        .with_columns([
            (
                pl.col("complete_zero_weeks_after_last_sale")
                >= SENSITIVITY_W
            ).alias("event_observed_w8"),

            pl.when(
                pl.col("complete_zero_weeks_after_last_sale")
                >= SENSITIVITY_W
            )
            .then(
                pl.col("last_positive_sales_week")
                + pl.duration(days=7)
            )
            .otherwise(None)
            .alias("death_week_w8"),
        ])
        .with_columns([
            pl.when(pl.col("event_observed_w8"))
            .then(pl.lit("death_observed"))
            .otherwise(pl.lit("right_censored"))
            .alias("lifecycle_status_w8"),

            pl.when(pl.col("event_observed_w8"))
            .then(
                (
                    (
                        pl.col("last_positive_sales_week")
                        + pl.duration(days=7)
                        - pl.col("birth_week")
                    ).dt.total_days() // 7
                ).cast(pl.Int32)
            )
            .otherwise(None)
            .alias("death_age_week_w8"),
        ])
    )

    lifecycle_labels_df = lifecycle_labels.collect()

    #Create final clean label table
    print("\nPreparing final lifecycle label table...")

    lifecycle_final = (
        lifecycle_labels_df
        .rename({
            "censor_age_week": "observation_end_age_week"
        })
        .with_columns([
            pl.when(pl.col("event_observed_w4"))
            .then(None)
            .otherwise(pl.col("observation_end_age_week"))
            .cast(pl.Int32)
            .alias("censor_age_week_w4"),

            pl.when(pl.col("event_observed_w8"))
            .then(None)
            .otherwise(pl.col("observation_end_age_week"))
            .cast(pl.Int32)
            .alias("censor_age_week_w8"),
        ])
        .select([
            "article_id",
            "birth_date",
            "birth_week",
            "last_positive_sales_week",
            "complete_zero_weeks_after_last_sale",
            "lifecycle_status_w4",
            "event_observed_w4",
            "death_week_w4",
            "death_age_week_w4",
            "censor_age_week_w4",
            "lifecycle_status_w8",
            "event_observed_w8",
            "death_week_w8",
            "death_age_week_w8",
            "censor_age_week_w8",
            "censor_date",
            "observation_end_age_week",
        ])
    )

    #Save temporary output
    print("\nSaving lifecycle labels...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lifecycle_final.write_parquet(OUTPUT_PATH)

    print("Saved:", OUTPUT_PATH)

    #Validate lifecycle outcomes
    print("\nValidating lifecycle outcomes...")

    summary = lifecycle_final.select([
        pl.len().alias("total_articles"),
        pl.col("event_observed_w4").sum().alias("deaths_confirmed_w4"),
        (~pl.col("event_observed_w4")).sum().alias("right_censored_w4"),
        pl.col("event_observed_w8").sum().alias("deaths_confirmed_w8"),
        (~pl.col("event_observed_w8")).sum().alias("right_censored_w8"),
    ])

    print(summary)

    values = summary.row(0, named=True)

    assert values["total_articles"] == EXPECTED_ARTICLES
    assert values["deaths_confirmed_w4"] == EXPECTED_DEATHS_W4
    assert values["right_censored_w4"] == EXPECTED_CENSORED_W4
    assert values["deaths_confirmed_w8"] == EXPECTED_DEATHS_W8
    assert values["right_censored_w8"] == EXPECTED_CENSORED_W8

    print("\nLIFECYCLE LABEL GENERATION COMPLETED")
    print("All expected-value checks passed.")
    print("Rows:", lifecycle_final.height)
    print("Columns:", lifecycle_final.columns)


if __name__ == "__main__":
    main()