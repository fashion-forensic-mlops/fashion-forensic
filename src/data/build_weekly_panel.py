from pathlib import Path
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEEKLY_SALES_PATH = (
    PROJECT_ROOT / "data/interim/weekly_sales_observed.parquet"
)

BIRTH_DATES_PATH = (
    PROJECT_ROOT / "data/interim/article_birth_dates.parquet"
)

BOUNDARIES_PATH = (
    PROJECT_ROOT / "data/interim/observation_boundaries.parquet"
)

#Weekly panel output
OUTPUT_PATH = (
    PROJECT_ROOT / "data/processed/weekly_panel.parquet"
)


def validate_inputs():
    required_paths = [
        WEEKLY_SALES_PATH,
        BIRTH_DATES_PATH,
        BOUNDARIES_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )


def main():
    print("=" * 60)
    print("M2 - WEEKLY PANEL CONSTRUCTION")
    print("=" * 60)

    validate_inputs()

    #Load prepared datasets
    print("\nLoading weekly sales and article birth dates...")

    weekly_sales = pl.scan_parquet(WEEKLY_SALES_PATH)
    birth_dates = pl.scan_parquet(BIRTH_DATES_PATH)

    #Load observation boundaries
    print("\nLoading observation boundaries...")

    boundaries = pl.read_parquet(BOUNDARIES_PATH)

    first_week = boundaries["first_week"][0]
    cutoff_week = boundaries["cutoff_week"][0]

    print("First observed week:", first_week)
    print("Final observed week:", cutoff_week)

    #Verify boundaries against the weekly sales data
    weekly_sales_boundaries = (
        weekly_sales
        .select([
            pl.col("week_start").min().alias("first_week"),
            pl.col("week_start").max().alias("cutoff_week"),
        ])
        .collect()
    )

    assert weekly_sales_boundaries["first_week"][0] == first_week
    assert weekly_sales_boundaries["cutoff_week"][0] == cutoff_week

    #Create continuous weekly calendar
    print("\nCreating weekly calendar...")

    calendar = pl.DataFrame({
        "week_start": pl.date_range(
            start=first_week,
            end=cutoff_week,
            interval="1w",
            eager=True,
        )
    })

    print("Total calendar weeks:", calendar.height)

    #Determine each article's birth week
    print("\nCalculating article birth weeks...")

    article_births = birth_dates.with_columns(
        pl.col("birth_date")
        .dt.truncate("1w")
        .alias("birth_week")
    )

    #Create article-specific weekly timelines
    print("\nGenerating article timelines...")

    article_calendar = (
        article_births
        .join(calendar.lazy(), how="cross")
        .filter(
            pl.col("week_start") >= pl.col("birth_week")
        )
    )

    #Attach sales and fill missing weeks with zero
    print("\nAttaching observed sales...")

    weekly_panel = (
        article_calendar
        .join(
            weekly_sales,
            on=["article_id", "week_start"],
            how="left",
        )
        .with_columns(
            pl.col("units_sold")
            .fill_null(0)
            .cast(pl.UInt32)
        )
    )

    #Calculate product age in weeks
    print("\nCalculating product age...")

    weekly_panel = (
        weekly_panel
        .with_columns(
            (
                (pl.col("week_start") - pl.col("birth_week"))
                .dt.total_days()
                // 7
            )
            .cast(pl.Int32)
            .alias("age_week")
        )
        .select([
            "article_id",
            "birth_date",
            "birth_week",
            "week_start",
            "age_week",
            "units_sold",
        ])
    )

    #Save generated panel
    print("\nSaving weekly panel...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly_panel.sink_parquet(OUTPUT_PATH)

    print("Saved:", OUTPUT_PATH)

    #Validate generated dataset
    print("\nValidating generated panel...")

    result = pl.scan_parquet(OUTPUT_PATH)

    summary = result.select([
        pl.len().alias("total_rows"),
        pl.col("article_id").n_unique().alias("unique_articles"),
        pl.col("units_sold").sum().alias("total_units_sold"),
        pl.col("units_sold")
        .filter(pl.col("units_sold") == 0)
        .len()
        .alias("zero_sales_weeks"),
        pl.col("age_week").min().alias("minimum_age_week"),
        pl.col("age_week").max().alias("maximum_age_week"),
    ]).collect()

    print(summary)

    values = summary.row(0, named=True)

    assert values["total_rows"] == 7_371_317
    assert values["unique_articles"] == 104_547
    assert values["total_units_sold"] == 31_788_324
    assert values["zero_sales_weeks"] == 5_160_418
    assert values["minimum_age_week"] == 0
    assert values["maximum_age_week"] == 105

    print("\nWEEKLY PANEL CONSTRUCTION COMPLETED")
    print("All expected-value checks passed.")


if __name__ == "__main__":
    main()