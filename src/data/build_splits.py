from datetime import date
from pathlib import Path

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTICLES_PATH = PROJECT_ROOT / "data/raw/articles.csv"
BIRTH_DATES_PATH = PROJECT_ROOT / "data/interim/article_birth_dates.parquet"
CONFIG_PATH = PROJECT_ROOT / "configs/splits.yaml"

ARTICLE_SPLITS_PATH = PROJECT_ROOT / "data/processed/article_splits.parquet"
PRODUCT_GROUP_SPLITS_PATH = (
    PROJECT_ROOT / "data/processed/product_group_splits.parquet"
)
SPLIT_SUMMARY_PATH = PROJECT_ROOT / "data/processed/split_summary.parquet"

EXPECTED_METADATA_ARTICLES = 105_542
EXPECTED_OBSERVED_ARTICLES = 104_547
EXPECTED_NO_SALES_ARTICLES = 995

EXPECTED_OBSERVED_GROUPS = 46_834
EXPECTED_UNOBSERVED_GROUPS = 390

SPLIT_ORDER = ["train", "validation", "test", "unassigned"]


def validate_inputs():
    required_paths = [
        ARTICLES_PATH,
        BIRTH_DATES_PATH,
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

    settings = config["splits"]

    train_end = date.fromisoformat(settings["train_end"])
    validation_end = date.fromisoformat(settings["validation_end"])

    if train_end >= validation_end:
        raise ValueError(
            "train_end must be earlier than validation_end."
        )

    return settings, train_end, validation_end


def assign_split(train_end, validation_end, unassigned_label):
    return (
        pl.when(pl.col("group_launch_date").is_null())
        .then(pl.lit(unassigned_label))
        .when(pl.col("group_launch_date") <= pl.lit(train_end))
        .then(pl.lit("train"))
        .when(
            pl.col("group_launch_date") <= pl.lit(validation_end)
        )
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )


def main():
    print("=" * 60)
    print("M2 - CHRONOLOGICAL PRODUCT GROUP SPLITTING")
    print("=" * 60)

    validate_inputs()

    settings, train_end, validation_end = load_config()

    article_column = settings["article_column"]
    group_column = settings["group_column"]
    birth_date_column = settings["birth_date_column"]
    unassigned_label = settings["unassigned_label"]

    print("\nSplit configuration:")
    print("Method:", settings["method"])
    print("Group column:", group_column)
    print("Train end:", train_end)
    print("Validation end:", validation_end)
    print("Unassigned label:", unassigned_label)

    #Load article metadata.
    print("\nLoading article metadata...")

    articles = (
        pl.read_csv(
            ARTICLES_PATH,
            columns=[article_column, group_column],
        )
        .with_columns([
            pl.col(article_column).cast(pl.Int64),
            pl.col(group_column).cast(pl.Int64),
        ])
    )

    #Load article birth dates.
    print("\nLoading article birth dates...")

    birth_dates = (
        pl.read_parquet(BIRTH_DATES_PATH)
        .select([
            pl.col(article_column).cast(pl.Int64),
            pl.col(birth_date_column).cast(pl.Date),
        ])
    )

    #Validate source tables.
    print("\nValidating source tables...")

    assert articles.height == EXPECTED_METADATA_ARTICLES
    assert articles[article_column].n_unique() == articles.height
    assert articles[group_column].null_count() == 0

    assert birth_dates.height == EXPECTED_OBSERVED_ARTICLES
    assert birth_dates[article_column].n_unique() == birth_dates.height
    assert birth_dates[birth_date_column].null_count() == 0

    unknown_birth_ids = (
        birth_dates
        .join(
            articles.select(article_column),
            on=article_column,
            how="anti",
        )
    )

    assert unknown_birth_ids.is_empty(), (
        "Birth-date table contains article IDs absent from metadata."
    )

    #Attach birth dates to metadata for observed articles.
    print("\nJoining article metadata with observed birth dates...")

    observed_articles = articles.join(
        birth_dates,
        on=article_column,
        how="inner",
    )

    assert observed_articles.height == EXPECTED_OBSERVED_ARTICLES

    #Derive one launch date per product group
    #The earliest observed article birth date represents the group launch
    print("\nCalculating product group launch dates...")

    observed_groups = (
        observed_articles
        .group_by(group_column)
        .agg([
            pl.col(birth_date_column)
            .min()
            .alias("group_launch_date"),

            pl.len().alias("article_count_with_sales"),
        ])
    )

    assert observed_groups.height == EXPECTED_OBSERVED_GROUPS

    #Count all metadata articles belonging to each product group
    all_group_counts = (
        articles
        .group_by(group_column)
        .agg(
            pl.len().alias("article_count_total")
        )
    )

    #Include groups without any observed sales
    print("\nBuilding complete product group table...")

    product_groups = (
        all_group_counts
        .join(
            observed_groups,
            on=group_column,
            how="left",
        )
        .with_columns([
            pl.col("article_count_with_sales")
            .fill_null(0)
            .cast(pl.Int64),

            pl.col("group_launch_date").cast(pl.Date),
        ])
        .with_columns(
            pl.col("article_count_total")
            .sub(pl.col("article_count_with_sales"))
            .alias("article_count_without_sales")
        )
        .with_columns(
            assign_split(
                train_end,
                validation_end,
                unassigned_label,
            )
        )
        .sort(group_column)
    )

    #Validate group assignments
    print("\nValidating product group assignments...")

    assert product_groups[group_column].n_unique() == product_groups.height

    observed_group_count = (
        product_groups
        .filter(pl.col("group_launch_date").is_not_null())
        .height
    )

    unobserved_group_count = (
        product_groups
        .filter(pl.col("group_launch_date").is_null())
        .height
    )

    assert observed_group_count == EXPECTED_OBSERVED_GROUPS
    assert unobserved_group_count == EXPECTED_UNOBSERVED_GROUPS

    assert (
        product_groups
        .filter(
            pl.col("group_launch_date").is_null()
            & (pl.col("split") != unassigned_label)
        )
        .is_empty()
    )

    #Expand each product group's split assignment to all its articles
    print("\nAssigning splits to individual articles...")

    article_splits = (
        articles
        .join(
            birth_dates,
            on=article_column,
            how="left",
        )
        .join(
            product_groups.select([
                group_column,
                "group_launch_date",
                "article_count_total",
                "article_count_with_sales",
                "article_count_without_sales",
                "split",
            ]),
            on=group_column,
            how="left",
        )
        .with_columns(
            pl.col(birth_date_column)
            .is_not_null()
            .alias("has_observed_sales")
        )
        .select([
            article_column,
            group_column,
            birth_date_column,
            "group_launch_date",
            "has_observed_sales",
            "split",
        ])
        .sort(article_column)
    )

    #Validate the final article-level assignments
    print("\nValidating article-level assignments...")

    assert article_splits.height == EXPECTED_METADATA_ARTICLES
    assert article_splits[article_column].n_unique() == article_splits.height
    assert article_splits["split"].null_count() == 0

    observed_article_count = (
        article_splits
        .filter(pl.col("has_observed_sales"))
        .height
    )

    no_sales_article_count = (
        article_splits
        .filter(~pl.col("has_observed_sales"))
        .height
    )

    assert observed_article_count == EXPECTED_OBSERVED_ARTICLES
    assert no_sales_article_count == EXPECTED_NO_SALES_ARTICLES

    assert (
        article_splits
        .filter(
            pl.col("has_observed_sales")
            & (pl.col("split") == unassigned_label)
        )
        .is_empty()
    )

    assert (
        article_splits
        .filter(
            (~pl.col("has_observed_sales"))
            & (pl.col("split") != unassigned_label)
        )
        .height
        == 542
    )

    assert (
        article_splits
        .filter(pl.col("split") == unassigned_label)
        .height
        == 453
    )

    #Confirm that every product group has exactly one split
    split_leakage = (
        article_splits
        .group_by(group_column)
        .agg(
            pl.col("split").n_unique().alias("split_count")
        )
        .filter(pl.col("split_count") != 1)
    )

    assert split_leakage.is_empty(), (
        "A product group has articles assigned to multiple splits."
    )

    #Build an auditable summary table
    print("\nGenerating split summary...")

    article_summary = (
        article_splits
        .group_by("split")
        .agg([
            pl.len().alias("article_count"),
            pl.col("has_observed_sales")
            .sum()
            .alias("articles_with_sales"),
            (~pl.col("has_observed_sales"))
            .sum()
            .alias("articles_without_sales"),
        ])
    )

    group_summary = (
        product_groups
        .group_by("split")
        .agg(
            pl.len().alias("product_group_count")
        )
    )

    split_summary = (
        group_summary
        .join(
            article_summary,
            on="split",
            how="inner",
        )
        .sort("split")
    )

    #Save outputs
    print("\nSaving split datasets...")

    ARTICLE_SPLITS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    article_splits.write_parquet(ARTICLE_SPLITS_PATH)
    product_groups.write_parquet(PRODUCT_GROUP_SPLITS_PATH)
    split_summary.write_parquet(SPLIT_SUMMARY_PATH)

    print("Saved:", ARTICLE_SPLITS_PATH)
    print("Saved:", PRODUCT_GROUP_SPLITS_PATH)
    print("Saved:", SPLIT_SUMMARY_PATH)

    #Display final results
    print("\n===== PRODUCT GROUP SPLITS =====")
    print(
        product_groups
        .group_by("split")
        .agg(pl.len().alias("product_groups"))
        .sort("split")
    )

    print("\n===== ARTICLE SPLIT SUMMARY =====")
    print(split_summary)

    print("\n===== FINAL VALIDATION =====")
    print("Metadata articles:", article_splits.height)
    print("Articles with observed sales:", observed_article_count)
    print("Articles without observed sales:", no_sales_article_count)
    print("Observed product groups:", observed_group_count)
    print("Unobserved product groups:", unobserved_group_count)
    print("Product group split leakage:", split_leakage.height)

    print("\nPRODUCT GROUP SPLITTING COMPLETED")
    print("All validation checks passed.")


if __name__ == "__main__":
    main()