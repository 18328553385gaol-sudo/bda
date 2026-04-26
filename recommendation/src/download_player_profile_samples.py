import argparse
import os
from pathlib import Path

from google.cloud import bigquery


DEFAULT_LOCAL_DIR = "artifacts/player_profile"
DEFAULT_PROJECT = os.getenv("PLAYER_PROFILE_BQ_PROJECT", "big-data-488609")
DEFAULT_TABLES = {
    "player_profile": os.getenv("PLAYER_PROFILE_BQ_TABLE_PROFILE", "feature.player_profile"),
    "market_value_history": os.getenv("PLAYER_PROFILE_BQ_TABLE_VALUE_HISTORY", "generaldata.market_value_history"),
    "player_heatmap": os.getenv("PLAYER_PROFILE_BQ_TABLE_HEATMAP", "feature.player_heatmap"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download player profile sample parquet files from BigQuery."
    )
    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT),
        help="Google Cloud project ID for BigQuery.",
    )
    parser.add_argument(
        "--local-dir",
        default=os.getenv("PLAYER_PROFILE_LOCAL_DIR", DEFAULT_LOCAL_DIR),
        help="Local output directory for downloaded parquet files.",
    )
    parser.add_argument(
        "--player-keys",
        nargs="+",
        help="Optional player_key values to filter all tables.",
    )
    parser.add_argument(
        "--player-name",
        help="Optional player_name filter for lookup or export.",
    )
    parser.add_argument(
        "--find-player",
        action="store_true",
        help="Search candidate players by name and print matching player_key values without exporting parquet files.",
    )
    parser.add_argument(
        "--find-complete-players",
        action="store_true",
        help="List players that exist in profile, market value history, and heatmap tables.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Row limit for each exported table when no player_keys are provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated SQL without running queries.",
    )
    return parser.parse_args()


def quote_identifier(project: str, table_path: str) -> str:
    return f"`{project}.{table_path}`"


def quote_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def build_where_clause(table_alias: str, player_keys: list[str] | None, player_name: str | None, export_name: str) -> str:
    filters = []

    if player_keys:
        quoted_keys = ", ".join(quote_string(key) for key in player_keys)
        filters.append(f"{table_alias}.player_key IN ({quoted_keys})")

    if player_name and export_name == "player_profile":
        filters.append(f"{table_alias}.player_name = {quote_string(player_name)}")

    if not filters:
        return ""

    return "WHERE " + " AND ".join(filters)


def build_player_lookup_query(project: str, player_name: str, limit: int) -> str:
    qualified_table = quote_identifier(project, DEFAULT_TABLES["player_profile"])
    name_value = player_name.strip().lower()
    return (
        f"SELECT DISTINCT t.player_key, t.player_name, t.current_club, t.position\n"
        f"FROM {qualified_table} AS t\n"
        f"WHERE LOWER(t.player_name) LIKE {quote_string(f'%{name_value}%')}\n"
        f"ORDER BY t.player_name, t.current_club, t.player_key\n"
        f"LIMIT {limit}"
    )


def build_complete_players_query(project: str, player_name: str | None, limit: int) -> str:
    profile_table = quote_identifier(project, DEFAULT_TABLES["player_profile"])
    value_table = quote_identifier(project, DEFAULT_TABLES["market_value_history"])
    heatmap_table = quote_identifier(project, DEFAULT_TABLES["player_heatmap"])

    name_filter = ""
    if player_name:
        name_value = player_name.strip().lower()
        name_filter = f"\nWHERE LOWER(p.player_name) LIKE {quote_string(f'%{name_value}%')}"

    return (
        "WITH value_counts AS (\n"
        f"    SELECT player_key, COUNT(*) AS value_history_rows\n"
        f"    FROM {value_table}\n"
        f"    GROUP BY player_key\n"
        "),\n"
        "heatmap_counts AS (\n"
        f"    SELECT player_key, COUNT(*) AS heatmap_rows\n"
        f"    FROM {heatmap_table}\n"
        f"    GROUP BY player_key\n"
        ")\n"
        "SELECT DISTINCT\n"
        "    p.player_key,\n"
        "    p.player_name,\n"
        "    p.current_club,\n"
        "    p.position,\n"
        "    vc.value_history_rows,\n"
        "    hc.heatmap_rows\n"
        f"FROM {profile_table} AS p\n"
        "JOIN value_counts AS vc ON p.player_key = vc.player_key\n"
        "JOIN heatmap_counts AS hc ON p.player_key = hc.player_key"
        f"{name_filter}\n"
        "ORDER BY vc.value_history_rows DESC, hc.heatmap_rows DESC, p.player_name, p.player_key\n"
        f"LIMIT {limit}"
    )


def build_query(project: str, table_path: str, export_name: str, player_keys: list[str] | None, player_name: str | None, limit: int) -> str:
    qualified_table = quote_identifier(project, table_path)
    where_clause = build_where_clause("t", player_keys, player_name, export_name)
    limit_clause = "" if player_keys else f"\nLIMIT {limit}"

    if export_name == "market_value_history":
        order_clause = "\nORDER BY t.player_key, SAFE_CAST(t.date AS DATE), t.date"
    elif export_name == "player_heatmap":
        order_clause = "\nORDER BY t.player_key, t.event_type_group, t.grid_y, t.grid_x"
    else:
        order_clause = "\nORDER BY t.player_name, t.player_key"

    return (
        f"SELECT *\n"
        f"FROM {qualified_table} AS t\n"
        f"{where_clause}"
        f"{order_clause}"
        f"{limit_clause}"
    )


def run_query_to_parquet(client: bigquery.Client, export_name: str, query: str, output_dir: Path):
    print(f"[{export_name}] Running query...")
    dataframe = client.query(query).to_dataframe(create_bqstorage_client=False)

    output_path = output_dir / f"{export_name}.parquet"
    dataframe.to_parquet(output_path, index=False)

    print(f"[{export_name}] Rows: {len(dataframe)}")
    print(f"[{export_name}] Saved to: {output_path}")
    print("-" * 60)


def lookup_players(client: bigquery.Client, project: str, player_name: str, limit: int, dry_run: bool):
    query = build_player_lookup_query(project=project, player_name=player_name, limit=limit)
    print("[find-player] SQL:")
    print(query)
    print("-" * 60)

    if dry_run:
        return

    dataframe = client.query(query).to_dataframe(create_bqstorage_client=False)
    if dataframe.empty:
        print("[find-player] No matching players found.")
        return

    print("[find-player] Candidates:")
    for _, row in dataframe.iterrows():
        print(
            f"player_key={row['player_key']} | "
            f"player_name={row['player_name']} | "
            f"current_club={row.get('current_club', 'N/A')} | "
            f"position={row.get('position', 'N/A')}"
        )
    print("-" * 60)


def lookup_complete_players(client: bigquery.Client, project: str, player_name: str | None, limit: int, dry_run: bool):
    query = build_complete_players_query(project=project, player_name=player_name, limit=limit)
    print("[find-complete-players] SQL:")
    print(query)
    print("-" * 60)

    if dry_run:
        return

    dataframe = client.query(query).to_dataframe(create_bqstorage_client=False)
    if dataframe.empty:
        print("[find-complete-players] No matching players found.")
        return

    print("[find-complete-players] Candidates:")
    for _, row in dataframe.iterrows():
        print(
            f"player_key={row['player_key']} | "
            f"player_name={row['player_name']} | "
            f"current_club={row.get('current_club', 'N/A')} | "
            f"position={row.get('position', 'N/A')} | "
            f"value_history_rows={row.get('value_history_rows', 0)} | "
            f"heatmap_rows={row.get('heatmap_rows', 0)}"
        )
    print("-" * 60)


def main():
    args = parse_args()

    if not args.project:
        raise ValueError("Missing GCP project ID. Use --project or set GOOGLE_CLOUD_PROJECT.")

    client = bigquery.Client(project=args.project)

    if args.find_player:
        if not args.player_name:
            raise ValueError("--find-player requires --player-name.")
        lookup_players(
            client=client,
            project=args.project,
            player_name=args.player_name,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        return

    if args.find_complete_players:
        lookup_complete_players(
            client=client,
            project=args.project,
            player_name=args.player_name,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        return

    output_dir = Path(args.local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for export_name, table_path in DEFAULT_TABLES.items():
        query = build_query(
            project=args.project,
            table_path=table_path,
            export_name=export_name,
            player_keys=args.player_keys,
            player_name=args.player_name,
            limit=args.limit,
        )

        print(f"[{export_name}] SQL:")
        print(query)
        print("-" * 60)

        if args.dry_run:
            continue

        run_query_to_parquet(
            client=client,
            export_name=export_name,
            query=query,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
