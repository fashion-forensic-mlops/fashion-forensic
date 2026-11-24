from datetime import timedelta
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"

TRANSACTIONS_PATH = RAW_DATA_DIR / "transactions_train.csv"

WEEKLY_SALES_PATH = INTERIM_DATA_DIR / "weekly_sales_observed.parquet"
BIRTH_DATES_PATH = INTERIM_DATA_DIR / "article_birth_dates.parquet"
BOUNDARIES_PATH = INTERIM_DATA_DIR / "observation_boundaries.parquet"

def validate_inputs():
    if not TRANSACTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {TRANSACTIONS_PATH}"
        )

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_transactions():
    print("Loading transaction dataset...")

    transactions = pl.scan_csv(
        TRANSACTIONS_PATH,
        infer_schema_length=10000,
    )

    print("Transaction schema:")
    print(transactions.collect_schema())

    return transactions


def prepare_transaction_dates(transactions):
    print("\nParsing transaction dates...")

    return transactions.with_columns(
        pl.col("t_dat")
        .str.to_date("%Y-%m-%d")
        .alias("transaction_date")
    )

def build_weekly_sales(transactions_typed):
    print("\nAggregating weekly article sales...")

    return (
        transactions_typed
        .with_columns(
            pl.col("transaction_date")
            .dt.truncate("1w")
            .alias("week_start")
        )
        .group_by(["article_id", "week_start"])
        .agg(
            pl.len().alias("units_sold")
        )
        .sort(["article_id", "week_start"])
    )

def build_birth_dates(transactions_typed):
    print("\nCalculating first recorded sale per article...")

    return (
        transactions_typed
        .group_by("article_id")
        .agg(
            pl.col("transaction_date")
            .min()
            .alias("birth_date")
        )
        .sort("article_id")
    )

def build_observation_boundaries(transactions_typed):
    print("\nDetermining transaction observation boundaries...")

    dates = (
        transactions_typed
        .select([
            pl.col("transaction_date").min().alias("first_transaction_date"),
            pl.col("transaction_date").max().alias("last_transaction_date"),
        ])
        .collect(engine="streaming")
    )

    first_date = dates["first_transaction_date"][0]
    last_date = dates["last_transaction_date"][0]

    cutoff_week = last_date - timedelta(days=last_date.weekday())
    week_end = cutoff_week + timedelta(days=6)

    if last_date >= week_end:
        last_complete_week = cutoff_week
    else:
        last_complete_week = cutoff_week - timedelta(days=7)

    first_week = first_date - timedelta(days=first_date.weekday())

    boundaries = pl.DataFrame({
        "first_transaction_date": [first_date],
        "last_transaction_date": [last_date],
        "first_week": [first_week],
        "cutoff_week": [cutoff_week],
        "last_complete_week": [last_complete_week],
    })

    print(boundaries)

    return boundaries

def save_outputs(
    weekly_sales,
    birth_dates,
    boundaries,
):
    print("\nSaving weekly sales...")
    weekly_sales.sink_parquet(WEEKLY_SALES_PATH)
    print("Saved:", WEEKLY_SALES_PATH)

    print("\nSaving article birth dates...")
    birth_dates.sink_parquet(BIRTH_DATES_PATH)
    print("Saved:", BIRTH_DATES_PATH)

    print("\nSaving observation boundaries...")
    boundaries.write_parquet(BOUNDARIES_PATH)
    print("Saved:", BOUNDARIES_PATH)

def main():
    print("=" * 60)
    print("M2 - WEEKLY DATA PREPARATION")
    print("=" * 60)

    validate_inputs()

    transactions = load_transactions()

    transactions_typed = prepare_transaction_dates(transactions)

    weekly_sales = build_weekly_sales(transactions_typed)

    birth_dates = build_birth_dates(transactions_typed)

    boundaries = build_observation_boundaries(transactions_typed)

    save_outputs(
        weekly_sales,
        birth_dates,
        boundaries,
    )

    print("\n" + "=" * 60)
    print("WEEKLY DATA PREPARATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()