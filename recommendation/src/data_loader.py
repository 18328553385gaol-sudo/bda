import os

import pandas as pd
from google.cloud import bigquery

from config import (
    DEFAULT_FEATURE_SOURCE_MODE,
    PROCESSED_FEATURES_PATH,
    TRAIN_BQ_FEATURE_TABLE,
    TRAIN_BQ_PROJECT,
)


def load_local_features(path=PROCESSED_FEATURES_PATH):
    return pd.read_parquet(path)


def load_bigquery_features(
    project: str = TRAIN_BQ_PROJECT,
    table: str = TRAIN_BQ_FEATURE_TABLE,
    limit: int | None = None,
):
    if not project:
        raise ValueError("TRAIN_BQ_PROJECT is required for BigQuery feature loading.")
    if not table:
        raise ValueError("TRAIN_BQ_FEATURE_TABLE is required for BigQuery feature loading.")

    client = bigquery.Client(project=project)
    limit_clause = f"\nLIMIT {limit}" if limit is not None else ""
    query = (
        "SELECT *\n"
        f"FROM `{project}.{table}`"
        f"{limit_clause}"
    )
    return client.query(query).to_dataframe(create_bqstorage_client=False)


def load_features(
    source_mode: str | None = None,
    local_path: str = PROCESSED_FEATURES_PATH,
    bq_project: str = TRAIN_BQ_PROJECT,
    bq_table: str = TRAIN_BQ_FEATURE_TABLE,
    bq_limit: int | None = None,
):
    mode = (source_mode or DEFAULT_FEATURE_SOURCE_MODE or "local").lower()

    if mode == "local":
        return load_local_features(path=local_path)
    if mode == "bigquery":
        return load_bigquery_features(project=bq_project, table=bq_table, limit=bq_limit)

    raise ValueError(f"Unsupported feature source mode: {mode}")
