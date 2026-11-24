from datetime import date
from pathlib import Path

import polars as pl
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PANEL_PATH = PROJECT_ROOT / "data/processed/weekly_panel.parquet"
SPLITS_PATH = PROJECT_ROOT / "data/processed/article_splits.parquet"
CONFIG_PATH = PROJECT_ROOT / "configs/forecasting.yaml"

DATASET_PATH = PROJECT_ROOT / "data/processed/forecasting_dataset.parquet"
SUMMARY_PATH = (
    PROJECT_ROOT / "data/processed/forecasting_eligibility_summary.parquet"
)

EXPECTED_METADATA_ARTICLES = 105_542
EXPECTED_OBSERVED_ARTICLES = 104_547
EXPECTED_ELIGIBLE_ARTICLES = 97_253

EXPECTED_ELIGIBLE_BY_SPLIT = {
    "train": 76_997,
    "validation": 13_219,
    "test": 7_037,
    "unassigned": 0,
}


def validate_inputs():
    required_paths = [
        PANEL_PATH,
        SPLITS_PATH,
        CONFIG_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    settings = config["forecasting"]

    horizon = int(settings["horizon"])

    target_start = int(settings["target"]["start_age_week"])
    target_end = int(settings["target"]["end_age_week"])

    history_weeks = int(
        settings["input"]["mode_b_history_weeks"]
    )

    history_start = int(
        settings["modes"]["mode_b"]["history_start_age_week"]
    )
    history_end = int(
        settings["modes"]["mode_b"]["history_end_age_week"]
    )

    last_complete_week = date.fromisoformat(
        settings["observation"]["last_complete_week"]
    )

    if target_end - target_start + 1 != horizon:
        raise ValueError(
            "Target age range does not match forecast horizon."
        )

    if history_end - history_start + 1 != history_weeks:
        raise ValueError(
            "Mode B history range does not match configured history weeks."
        )

    if target_start != history_end + 1:
        raise ValueError(
            "Target window must immediately follow Mode B input history."
        )

    if not settings["eligibility"]["require_complete_target_window"]:
        raise ValueError(
            "This builder requires complete target windows."
        )

    if not settings["eligibility"]["retain_ineligible_articles"]:
        raise ValueError(
            "This builder is designed to retain ineligible articles."
        )

    return (
        settings,
        horizon,
        target_start,
        target_end,
        history_start,
        history_end,
        last_complete_week,
    )


def main():
    print("=" * 65)
    print("M2 - FORECASTING DATASET GENERATION")
    print("=" * 65)

    validate_inputs()

    (
        settings,
        horizon,
        target_start,
        target_end,
        history_start,
        history_end,
        last_complete_week,
    ) = load_config()

    print("\nForecasting configuration:")
    print("Horizon:", horizon)
    print(f"Target ages: {target_start}–{target_end}")
    print(f"Mode B history ages: {history_start}–{history_end}")
    print("Last complete week:", last_complete_week)

    # ---------------------------------------------------------
    # 1. Load article-level split assignments.
    # ---------------------------------------------------------

    print("\nLoading article split assignments...")

    articles = pl.read_parquet(SPLITS_PATH)

    assert articles.height == EXPECTED_METADATA_ARTICLES
    assert articles["article_id"].n_unique() == articles.height
    assert articles["article_id"].null_count() == 0

    assert articles["has_observed_sales"].sum() == (
        EXPECTED_OBSERVED_ARTICLES
    )

    print("Metadata articles:", articles.height)
    print(
        "Articles with observed sales:",
        articles["has_observed_sales"].sum(),
    )

    # ---------------------------------------------------------
    # 2. Aggregate the required weekly input and target values.
    # ---------------------------------------------------------

    print("\nReading weekly panel and extracting age-based values...")

    panel = pl.scan_parquet(PANEL_PATH)

    expressions = [
        pl.col("birth_week").first().alias("birth_week"),

        pl.col("age_week")
        .filter(
            pl.col("age_week").is_between(
                history_start, history_end
            )
        )
        .n_unique()
        .alias("mode_b_input_weeks_available"),

        pl.col("age_week")
        .filter(
            pl.col("age_week").is_between(
                target_start, target_end
            )
        )
        .n_unique()
        .alias("target_weeks_available"),

        pl.col("week_start")
        .filter(pl.col("age_week") == target_end)
        .first()
        .alias("target_end_week_start"),
    ]

    # Mode B input values: product ages 0–3.
    for age in range(history_start, history_end + 1):
        expressions.append(
            pl.col("units_sold")
            .filter(pl.col("age_week") == age)
            .first()
            .alias(f"mode_b_input_units_age_{age}")
        )

    # Shared target values: product ages 4–9.
    for age in range(target_start, target_end + 1):
        expressions.append(
            pl.col("units_sold")
            .filter(pl.col("age_week") == age)
            .first()
            .alias(f"target_units_age_{age}")
        )

    weekly_features = (
        panel
        .group_by("article_id")
        .agg(expressions)
        .collect()
    )

    print("Articles represented in weekly panel:", weekly_features.height)

    assert weekly_features["article_id"].n_unique() == (
        weekly_features.height
    )

    assert weekly_features.height == EXPECTED_OBSERVED_ARTICLES

    # ---------------------------------------------------------
    # 3. Join weekly features onto all metadata articles.
    # ---------------------------------------------------------

    print("\nJoining weekly features onto article metadata...")

    dataset = articles.join(
        weekly_features,
        on="article_id",
        how="left",
        validate="1:1",
    )

    assert dataset.height == EXPECTED_METADATA_ARTICLES

    # ---------------------------------------------------------
    # 4. Determine forecasting eligibility.
    # ---------------------------------------------------------

    print("\nCalculating forecasting eligibility...")

    dataset = dataset.with_columns([
        (
            pl.col("has_observed_sales")
            & (
                pl.col("mode_b_input_weeks_available")
                == (history_end - history_start + 1)
            )
        ).alias("mode_b_input_complete"),

        (
            pl.col("has_observed_sales")
            & (
                pl.col("target_weeks_available") == horizon
            )
            & (
                pl.col("target_end_week_start")
                <= pl.lit(last_complete_week)
            )
        ).alias("target_window_complete"),
    ])

    dataset = dataset.with_columns(
        (
            pl.col("mode_b_input_complete")
            & pl.col("target_window_complete")
        ).alias("forecast_eligible")
    )

    dataset = dataset.with_columns(
        pl.when(~pl.col("has_observed_sales"))
        .then(pl.lit("no_observed_sales"))
        .when(~pl.col("mode_b_input_complete"))
        .then(pl.lit("incomplete_input_history"))
        .when(~pl.col("target_window_complete"))
        .then(pl.lit("incomplete_target_window"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("ineligibility_reason")
    )

    # ---------------------------------------------------------
    # 5. Remove target values from ineligible records.
    # ---------------------------------------------------------

    print("\nProtecting target values from incomplete windows...")

    target_columns = [
        f"target_units_age_{age}"
        for age in range(target_start, target_end + 1)
    ]

    dataset = dataset.with_columns([
        pl.when(pl.col("forecast_eligible"))
        .then(pl.col(column))
        .otherwise(None)
        .alias(column)
        for column in target_columns
    ])

    # ---------------------------------------------------------
    # 6. Validate the completed dataset.
    # ---------------------------------------------------------

    print("\nValidating forecasting dataset...")

    eligible_count = dataset.filter(
        pl.col("forecast_eligible")
    ).height

    ineligible_count = dataset.filter(
        ~pl.col("forecast_eligible")
    ).height

    observed_count = dataset.filter(
        pl.col("has_observed_sales")
    ).height

    no_sales_count = dataset.filter(
        ~pl.col("has_observed_sales")
    ).height

    assert dataset.height == EXPECTED_METADATA_ARTICLES
    assert observed_count == EXPECTED_OBSERVED_ARTICLES
    assert no_sales_count == (
        EXPECTED_METADATA_ARTICLES - EXPECTED_OBSERVED_ARTICLES
    )

    assert eligible_count == EXPECTED_ELIGIBLE_ARTICLES
    assert eligible_count + ineligible_count == dataset.height

    # Every eligible article must have all four Mode B inputs.
    assert dataset.filter(
        pl.col("forecast_eligible")
        & ~pl.col("mode_b_input_complete")
    ).height == 0

    # Every eligible article must have all six target values.
    for column in target_columns:
        assert dataset.filter(
            pl.col("forecast_eligible")
            & pl.col(column).is_null()
        ).height == 0

    # Ineligible articles must not retain target values.
    for column in target_columns:
        assert dataset.filter(
            ~pl.col("forecast_eligible")
            & pl.col(column).is_not_null()
        ).height == 0

    # Validate eligibility counts by split.
    split_eligibility = (
        dataset
        .group_by("split")
        .agg(
            pl.col("forecast_eligible")
            .sum()
            .alias("eligible_articles")
        )
    )

    actual_by_split = {
        row["split"]: row["eligible_articles"]
        for row in split_eligibility.to_dicts()
    }

    for split, expected in EXPECTED_ELIGIBLE_BY_SPLIT.items():
        actual = actual_by_split.get(split, 0)

        assert actual == expected, (
            f"Eligible article count mismatch for {split}: "
            f"expected {expected}, got {actual}"
        )

    # ---------------------------------------------------------
    # 7. Build split-level eligibility summary.
    # ---------------------------------------------------------

    print("\nBuilding split-level eligibility summary...")

    summary = (
        dataset
        .group_by("split")
        .agg([
            pl.len().alias("total_articles"),

            pl.col("has_observed_sales")
            .sum()
            .alias("articles_with_observed_sales"),

            pl.col("forecast_eligible")
            .sum()
            .alias("eligible_articles"),

            (~pl.col("forecast_eligible"))
            .sum()
            .alias("ineligible_articles"),
        ])
        .with_columns(
            pl.when(
                pl.col("articles_with_observed_sales") > 0
            )
            .then(
                pl.col("eligible_articles")
                / pl.col("articles_with_observed_sales")
                * 100
            )
            .otherwise(0.0)
            .alias("eligibility_coverage_pct")
        )
        .sort("split")
    )

    # ---------------------------------------------------------
    # 8. Select and order output columns.
    # ---------------------------------------------------------

    input_columns = [
        f"mode_b_input_units_age_{age}"
        for age in range(history_start, history_end + 1)
    ]

    output_columns = [
        "article_id",
        "product_code",
        "split",
        "birth_date",
        "birth_week",
        "has_observed_sales",
        "mode_b_input_weeks_available",
        "target_weeks_available",
        "target_end_week_start",
        "mode_b_input_complete",
        "target_window_complete",
        "forecast_eligible",
        "ineligibility_reason",
        *input_columns,
        *target_columns,
    ]

    dataset = dataset.select(output_columns).sort("article_id")

    # ---------------------------------------------------------
    # 9. Save reproducible Parquet outputs.
    # ---------------------------------------------------------

    print("\nSaving forecasting dataset...")

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset.write_parquet(
        DATASET_PATH,
        compression="zstd",
    )

    summary.write_parquet(
        SUMMARY_PATH,
        compression="zstd",
    )

    # ---------------------------------------------------------
    # 10. Print final report.
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("FORECASTING DATASET SUMMARY")
    print("=" * 65)

    print("Total metadata articles:", dataset.height)
    print("Articles with observed sales:", observed_count)
    print("Articles without observed sales:", no_sales_count)
    print("Forecast-eligible articles:", eligible_count)
    print("Ineligible articles:", ineligible_count)
    print(
        "Overall coverage among observed articles:",
        f"{eligible_count / observed_count * 100:.2f}%",
    )

    print("\nEligibility by split:")
    print(summary)

    print("\nSaved:")
    print(DATASET_PATH)
    print(SUMMARY_PATH)

    print("\nPASS: Forecasting dataset generated successfully.")


if __name__ == "__main__":
    main()
