from __future__ import annotations

import os

import pandas as pd
from google.cloud import bigquery


DEFAULT_BQ_PROJECT = os.getenv("PLAYER_PROFILE_BQ_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", "big-data-488609"))
DEFAULT_BQ_TABLES = {
    "player_profile": os.getenv("PLAYER_PROFILE_BQ_TABLE_PROFILE", "feature.player_profile"),
    "market_value_history": os.getenv("PLAYER_PROFILE_BQ_TABLE_VALUE_HISTORY", "generaldata.market_value_history"),
    "player_heatmap": os.getenv("PLAYER_PROFILE_BQ_TABLE_HEATMAP", "feature.player_heatmap"),
}


def _quote_identifier(project: str, table_name: str) -> str:
    return f"`{project}.{table_name}`"


def _quote_string(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def get_bigquery_config(overrides: dict | None = None) -> dict:
    config = {
        "project": DEFAULT_BQ_PROJECT,
        "tables": DEFAULT_BQ_TABLES.copy(),
    }

    if overrides:
        config["project"] = overrides.get("bq_project", config["project"])
        config["tables"].update(
            {
                "player_profile": overrides.get("bq_player_profile_table", config["tables"]["player_profile"]),
                "market_value_history": overrides.get("bq_market_value_history_table", config["tables"]["market_value_history"]),
                "player_heatmap": overrides.get("bq_player_heatmap_table", config["tables"]["player_heatmap"]),
            }
        )

    return config


def _run_query(project: str, query: str) -> pd.DataFrame:
    client = bigquery.Client(project=project)
    return client.query(query).to_dataframe(create_bqstorage_client=False)


def load_player_options_bq(config: dict) -> pd.DataFrame:
    project = config["project"]
    profile_table = _quote_identifier(project, config["tables"]["player_profile"])
    query = (
        "SELECT DISTINCT player_key, player_name, current_club\n"
        f"FROM {profile_table}\n"
        "ORDER BY player_name, current_club, player_key"
    )
    return _run_query(project, query)


def load_player_profile_bq(player_key: str, config: dict) -> pd.DataFrame:
    project = config["project"]
    table_name = _quote_identifier(project, config["tables"]["player_profile"])
    query = (
        "SELECT *\n"
        f"FROM {table_name}\n"
        f"WHERE player_key = {_quote_string(player_key)}"
    )
    return _run_query(project, query)


def load_market_value_history_bq(player_key: str, config: dict) -> pd.DataFrame:
    project = config["project"]
    table_name = _quote_identifier(project, config["tables"]["market_value_history"])
    query = (
        "SELECT *\n"
        f"FROM {table_name}\n"
        f"WHERE player_key = {_quote_string(player_key)}\n"
        "ORDER BY SAFE_CAST(date AS DATE), date"
    )
    return _run_query(project, query)


def load_player_heatmap_bq(player_key: str, config: dict) -> pd.DataFrame:
    project = config["project"]
    table_name = _quote_identifier(project, config["tables"]["player_heatmap"])
    query = (
        "SELECT *\n"
        f"FROM {table_name}\n"
        f"WHERE player_key = {_quote_string(player_key)}\n"
        "ORDER BY event_type_group, grid_y, grid_x"
    )
    return _run_query(project, query)
