import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from player_profile_repository import (
    get_bigquery_config,
    load_market_value_history_bq,
    load_player_heatmap_bq,
    load_player_options_bq,
    load_player_profile_bq,
)


PAGE_TITLE = "Player Profile Dashboard"
LOCAL_DATA_DIR = "artifacts/player_profile"
REQUIRED_TABLES = {
    "player_profile": "feature.player_profile",
    "market_value_history": "market_value_history",
    "player_heatmap": "feature.player_heatmap",
}

HEATMAP_GROUP_ALIASES = {
    "global": ["global", "overall", "all", "full"],
    "attack": ["attack", "attacking", "offense", "offensive"],
    "build": ["build", "build_up", "buildip", "playmaking", "organisation", "organization", "progression"],
    "defense": ["defense", "defence", "defending", "defensive", "defend" ],
}

PITCH_LINE_COLOR = "rgba(255,255,255,0.85)"
PITCH_BG_COLOR = "#0F5D3B"
HEATMAP_COLOR_SCALE = [
    [0.0, "#f7fbff"],
    [0.2, "#deebf7"],
    [0.4, "#9ecae1"],
    [0.6, "#4292c6"],
    [0.8, "#2171b5"],
    [1.0, "#08306b"],
]


st.set_page_config(page_title=PAGE_TITLE, page_icon=":bar_chart:", layout="wide")


def has_streamlit_secrets_file() -> bool:
    secret_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    ]
    return any(path.exists() for path in secret_paths)


def get_setting(name: str, default: str | None = None) -> str | None:
    if has_streamlit_secrets_file() and name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


def get_data_source_config() -> dict:
    return {
        "mode": (get_setting("PLAYER_PROFILE_DATA_MODE", "local") or "local").lower(),
        "local_dir": get_setting("PLAYER_PROFILE_LOCAL_DIR", LOCAL_DATA_DIR) or LOCAL_DATA_DIR,
        "gcs_bucket": get_setting("PLAYER_PROFILE_GCS_BUCKET"),
        "gcs_prefix": (get_setting("PLAYER_PROFILE_GCS_PREFIX", "") or "").strip("/"),
        "bq_project": get_setting("PLAYER_PROFILE_BQ_PROJECT", get_setting("GOOGLE_CLOUD_PROJECT", "big-data-488609")),
        "bq_player_profile_table": get_setting("PLAYER_PROFILE_BQ_TABLE_PROFILE", "feature.player_profile"),
        "bq_market_value_history_table": get_setting("PLAYER_PROFILE_BQ_TABLE_VALUE_HISTORY", "generaldata.market_value_history"),
        "bq_player_heatmap_table": get_setting("PLAYER_PROFILE_BQ_TABLE_HEATMAP", "feature.player_heatmap"),
    }


def get_storage_options() -> dict | None:
    token_path = get_setting("GOOGLE_APPLICATION_CREDENTIALS")
    if token_path:
        return {"token": token_path}
    return None


def build_table_path(table_name: str, config: dict) -> str:
    if config["mode"] == "bigquery":
        raise ValueError("BigQuery mode does not use parquet file paths.")
    if config["mode"] == "gcs":
        bucket = config["gcs_bucket"]
        if not bucket:
            raise ValueError("PLAYER_PROFILE_GCS_BUCKET is required when PLAYER_PROFILE_DATA_MODE=gcs.")
        prefix = config["gcs_prefix"]
        parts = [f"gs://{bucket}"]
        if prefix:
            parts.append(prefix)
        parts.append(f"{table_name}.parquet")
        return "/".join(parts)

    local_dir = Path(config["local_dir"])
    return str(local_dir / f"{table_name}.parquet")


@st.cache_data(show_spinner=False)
def load_table(table_name: str, config: dict) -> pd.DataFrame:
    table_path = build_table_path(table_name, config)
    storage_options = get_storage_options() if config["mode"] == "gcs" else None
    return pd.read_parquet(table_path, storage_options=storage_options)


@st.cache_data(show_spinner=False)
def load_player_options_bigquery(config: dict) -> pd.DataFrame:
    return load_player_options_bq(get_bigquery_config(config))


@st.cache_data(show_spinner=False)
def load_bigquery_player_bundle(player_key: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bq_config = get_bigquery_config(config)
    profile_df = load_player_profile_bq(player_key, bq_config)
    value_history_df = load_market_value_history_bq(player_key, bq_config)
    heatmap_df = load_player_heatmap_bq(player_key, bq_config)
    return profile_df, value_history_df, heatmap_df


def normalize_text(value) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def find_heatmap_group(df: pd.DataFrame, canonical_group: str) -> pd.DataFrame:
    if "event_type_group" not in df.columns:
        return pd.DataFrame()

    normalized_groups = df["event_type_group"].astype(str).map(normalize_text)
    aliases = {normalize_text(alias) for alias in HEATMAP_GROUP_ALIASES[canonical_group]}
    return df[normalized_groups.isin(aliases)].copy()


def is_goalkeeper(profile_row: pd.Series) -> bool:
    position = str(profile_row.get("position", "")).lower()
    return "goalkeeper" in position or position in {"gk", "keeper"}


def format_currency(value) -> str:
    if pd.isna(value):
        return "N/A"

    value = float(value)
    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        return f"EUR {value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"EUR {value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"EUR {value / 1_000:.1f}K"
    return f"EUR {value:,.0f}"


def format_delta(value) -> str:
    if pd.isna(value):
        return "N/A"
    value = float(value)
    sign = "+" if value > 0 else ""
    return f"{sign}{format_currency(value).replace('EUR ', '')}"


def format_display_value(value, empty: str = "N/A") -> str | int | float:
    if value is None:
        return empty
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return empty
        return ", ".join(str(item) for item in value.tolist())
    if isinstance(value, (list, tuple, set)):
        if len(value) == 0:
            return empty
        return ", ".join(str(item) for item in value)
    if pd.isna(value):
        return empty
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def safe_divide(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0:
        return pd.NA
    return float(numerator) / float(denominator)


def calculate_market_value_summary(history_df: pd.DataFrame) -> pd.Series:
    if history_df.empty:
        return pd.Series(dtype="object")

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")
    df = df.sort_values("date")

    latest_row = df.iloc[-1].copy()
    current_market_value = latest_row.get("market_value")
    peak_value = df["market_value"].max()
    peak_ratio = safe_divide(current_market_value, peak_value)

    latest_date = latest_row.get("date")

    def lookup_change(years_back: int):
        if pd.isna(latest_date):
            return pd.NA
        threshold = latest_date - pd.DateOffset(years=years_back)
        previous_rows = df[df["date"] <= threshold]
        if previous_rows.empty:
            return pd.NA
        previous_value = previous_rows.iloc[-1].get("market_value")
        if pd.isna(previous_value) or pd.isna(current_market_value):
            return pd.NA
        return float(current_market_value) - float(previous_value)

    value_change_1yr = lookup_change(1)
    value_change_2yr = lookup_change(2)

    if pd.isna(value_change_1yr):
        value_trend = "stable"
    elif value_change_1yr > 0:
        value_trend = "up"
    elif value_change_1yr < 0:
        value_trend = "down"
    else:
        value_trend = "stable"

    value_volatility = df["market_value"].std()
    if pd.isna(value_change_1yr):
        value_label = "insufficient_history"
        value_label_binary = "unknown"
    elif value_change_1yr > 0:
        value_label = "rising"
        value_label_binary = 1
    elif value_change_1yr < 0:
        value_label = "falling"
        value_label_binary = 0
    else:
        value_label = "stable"
        value_label_binary = 0

    summary = latest_row.copy()
    summary["current_market_value"] = current_market_value
    summary["peak_value"] = peak_value
    summary["peak_ratio"] = peak_ratio
    summary["value_change_1yr"] = value_change_1yr
    summary["value_change_2yr"] = value_change_2yr
    summary["value_trend"] = value_trend
    summary["value_volatility"] = value_volatility
    summary["value_label"] = value_label
    summary["value_label_binary"] = value_label_binary
    return summary


def draw_pitch_shapes(x_domain: tuple[float, float], y_domain: tuple[float, float]) -> list[dict]:
    x0, x1 = x_domain
    y0, y1 = y_domain
    width = x1 - x0
    height = y1 - y0

    def sx(value: float) -> float:
        return x0 + (value / 120.0) * width

    def sy(value: float) -> float:
        return y0 + (value / 80.0) * height

    line = {"color": PITCH_LINE_COLOR, "width": 1.4}
    shapes = [
        {"type": "rect", "xref": "paper", "yref": "paper", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "line": line},
        {"type": "line", "xref": "paper", "yref": "paper", "x0": sx(60), "y0": y0, "x1": sx(60), "y1": y1, "line": line},
        {"type": "circle", "xref": "paper", "yref": "paper", "x0": sx(50.85), "y0": sy(30.85), "x1": sx(69.15), "y1": sy(49.15), "line": line},
        {"type": "rect", "xref": "paper", "yref": "paper", "x0": sx(0), "y0": sy(18), "x1": sx(18), "y1": sy(62), "line": line},
        {"type": "rect", "xref": "paper", "yref": "paper", "x0": sx(102), "y0": sy(18), "x1": sx(120), "y1": sy(62), "line": line},
        {"type": "rect", "xref": "paper", "yref": "paper", "x0": sx(0), "y0": sy(30), "x1": sx(6), "y1": sy(50), "line": line},
        {"type": "rect", "xref": "paper", "yref": "paper", "x0": sx(114), "y0": sy(30), "x1": sx(120), "y1": sy(50), "line": line},
    ]
    return shapes


def prepare_heatmap_grid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    value_col = "event_weight" if "event_weight" in df.columns else "event_count"
    heatmap = (
        df.groupby(["grid_y", "grid_x"], as_index=False)[value_col]
        .sum()
        .pivot(index="grid_y", columns="grid_x", values=value_col)
        .sort_index(ascending=False)
    )
    return heatmap


def add_heatmap_trace(fig, df: pd.DataFrame, row: int, col: int, title: str):
    heatmap = prepare_heatmap_grid(df)

    if heatmap.empty:
        fig.add_annotation(
            text=f"No data for {title}",
            x=60,
            y=40,
            xref=f"x{'' if (row, col) == (1, 1) else (row - 1) * 2 + col}",
            yref=f"y{'' if (row, col) == (1, 1) else (row - 1) * 2 + col}",
            showarrow=False,
            font={"color": "white", "size": 14},
        )
        return

    fig.add_trace(
        go.Heatmap(
            z=heatmap.values,
            x=heatmap.columns.tolist(),
            y=heatmap.index.tolist(),
            colorscale=HEATMAP_COLOR_SCALE,
            showscale=False,
            hovertemplate="grid_x=%{x}<br>grid_y=%{y}<br>weight=%{z:.2f}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def build_heatmap_figure(player_heatmap: pd.DataFrame, goalkeeper_mode: bool) -> go.Figure:
    subplot_titles = ["Global Activity", "Defensive Activity"] if goalkeeper_mode else [
        "Global Activity",
        "Attacking Activity",
        "Build-up Activity",
        "Defensive Activity",
    ]
    rows = 1 if goalkeeper_mode else 2
    cols = 2

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, horizontal_spacing=0.08, vertical_spacing=0.12)

    add_heatmap_trace(fig, find_heatmap_group(player_heatmap, "global"), 1, 1, "Global Activity")
    if goalkeeper_mode:
        add_heatmap_trace(fig, find_heatmap_group(player_heatmap, "defense"), 1, 2, "Defensive Activity")
    else:
        add_heatmap_trace(fig, find_heatmap_group(player_heatmap, "attack"), 1, 2, "Attacking Activity")
        add_heatmap_trace(fig, find_heatmap_group(player_heatmap, "build"), 2, 1, "Build-up Activity")
        add_heatmap_trace(fig, find_heatmap_group(player_heatmap, "defense"), 2, 2, "Defensive Activity")

    fig.update_layout(
        height=460 if goalkeeper_mode else 820,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        paper_bgcolor="#0B1320",
        plot_bgcolor=PITCH_BG_COLOR,
        font={"color": "white"},
    )

    for axis_name in fig.layout:
        if axis_name.startswith("xaxis") or axis_name.startswith("yaxis"):
            axis = fig.layout[axis_name]
            axis.update(showgrid=False, zeroline=False, visible=False)

    shapes = []
    for axis_name in fig.layout:
        if axis_name.startswith("xaxis"):
            suffix = axis_name[5:]
            y_axis_name = f"yaxis{suffix}"
            x_axis = fig.layout[axis_name]
            y_axis = fig.layout[y_axis_name]
            shapes.extend(draw_pitch_shapes(tuple(x_axis.domain), tuple(y_axis.domain)))

    fig.update_layout(shapes=shapes)
    return fig


def show_data_source_help(config: dict):
    with st.sidebar:
        st.subheader("Data Source")
        st.caption(f"Current mode: `{config['mode']}`")
        if config["mode"] == "bigquery":
            st.code(f"{config['bq_project']}.{config['bq_player_profile_table']}")
            st.code(f"{config['bq_project']}.{config['bq_market_value_history_table']}")
            st.code(f"{config['bq_project']}.{config['bq_player_heatmap_table']}")
        elif config["mode"] == "local":
            st.code(f"{config['local_dir']}/player_profile.parquet")
            st.code(f"{config['local_dir']}/market_value_history.parquet")
            st.code(f"{config['local_dir']}/player_heatmap.parquet")
        else:
            st.code(f"gs://{config['gcs_bucket']}/{config['gcs_prefix']}/player_profile.parquet")
            st.code(f"gs://{config['gcs_bucket']}/{config['gcs_prefix']}/market_value_history.parquet")
            st.code(f"gs://{config['gcs_bucket']}/{config['gcs_prefix']}/player_heatmap.parquet")

        st.markdown(
            """
            Set these env vars or Streamlit secrets to switch data source:
            - `PLAYER_PROFILE_DATA_MODE=local|gcs|bigquery`
            - `PLAYER_PROFILE_LOCAL_DIR=artifacts/player_profile`
            - `PLAYER_PROFILE_GCS_BUCKET=<your-bucket>`
            - `PLAYER_PROFILE_GCS_PREFIX=<optional-prefix>`
            - `PLAYER_PROFILE_BQ_PROJECT=<your-project>`
            - `PLAYER_PROFILE_BQ_TABLE_PROFILE=feature.player_profile`
            - `PLAYER_PROFILE_BQ_TABLE_VALUE_HISTORY=generaldata.market_value_history`
            - `PLAYER_PROFILE_BQ_TABLE_HEATMAP=feature.player_heatmap`
            - `GOOGLE_APPLICATION_CREDENTIALS=<service-account-json>`
            """
        )


def render_profile_cards(profile_row: pd.Series):
    st.subheader("Player Profile")
    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)

    c1.metric("Player", format_display_value(profile_row.get("player_name")))
    c2.metric("Position", format_display_value(profile_row.get("position")))
    c3.metric("Age", format_display_value(profile_row.get("age")))
    c4.metric("Height", format_display_value(profile_row.get("height")))
    c5.metric("Preferred Foot", format_display_value(profile_row.get("preferred_foot")))
    c6.metric("Birth Country", format_display_value(profile_row.get("birth_country")))
    c7.metric("Citizenship", format_display_value(profile_row.get("citizenship")))
    c8.metric("Current Club", format_display_value(profile_row.get("current_club")))

    st.info(f"Career Stage: {format_display_value(profile_row.get('career_stage'))}")


def render_value_summary(summary_row: pd.Series):
    st.subheader("Market Value Summary")
    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)
    m7, m8 = st.columns(2)

    m1.metric("Current Market Value", format_currency(summary_row.get("current_market_value")))
    m2.metric("Peak Value", format_currency(summary_row.get("peak_value")))
    peak_ratio = summary_row.get("peak_ratio")
    m3.metric("Peak Ratio", f"{float(peak_ratio):.2f}" if pd.notna(peak_ratio) else "N/A")
    m4.metric("Value Change (1Y)", format_delta(summary_row.get("value_change_1yr")))
    m5.metric("Value Change (2Y)", format_delta(summary_row.get("value_change_2yr")))
    m6.metric("Trend", summary_row.get("value_trend", "N/A"))
    m7.metric("Value Label", summary_row.get("value_label", "N/A"))
    m8.metric("Binary Label", summary_row.get("value_label_binary", "N/A"))


def render_market_value_chart(history_df: pd.DataFrame):
    st.subheader("Market Value History")

    chart_df = history_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df = chart_df.sort_values("date")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["market_value"],
            mode="lines+markers",
            line={"color": "#2E86DE", "width": 3},
            marker={"size": 7, "color": "#154360"},
            customdata=chart_df[["club_name", "age"]].fillna("N/A").values,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Market Value: %{y:,.0f} EUR<br>"
                "Club: %{customdata[0]}<br>"
                "Age: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    if chart_df["market_value"].notna().any():
        peak_idx = chart_df["market_value"].idxmax()
        peak_row = chart_df.loc[peak_idx]
        fig.add_trace(
            go.Scatter(
                x=[peak_row["date"]],
                y=[peak_row["market_value"]],
                mode="markers",
                marker={"size": 14, "color": "#E74C3C", "symbol": "star"},
                name="Peak",
                hovertemplate=(
                    "Peak Date: %{x|%Y-%m-%d}<br>"
                    "Peak Value: %{y:,.0f} EUR<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis_title="Date",
        yaxis_title="Market Value (EUR)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def get_latest_summary_row(history_df: pd.DataFrame) -> pd.Series:
    summary_columns = [
        "current_market_value",
        "peak_value",
        "peak_ratio",
        "value_change_1yr",
        "value_change_2yr",
        "value_trend",
        "value_volatility",
        "value_label",
        "value_label_binary",
    ]
    if all(column in history_df.columns for column in summary_columns):
        sorted_df = history_df.copy()
        sorted_df["date"] = pd.to_datetime(sorted_df["date"], errors="coerce")
        sorted_df = sorted_df.sort_values("date")
        return sorted_df.iloc[-1]
    return calculate_market_value_summary(history_df)


def render_heatmaps(heatmap_df: pd.DataFrame, profile_row: pd.Series):
    st.subheader("Heatmap Overview")
    keeper_mode = is_goalkeeper(profile_row)
    fig = build_heatmap_figure(heatmap_df, keeper_mode)
    st.plotly_chart(fig, use_container_width=True)


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], table_name: str):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {', '.join(missing)}")


def main():
    st.title("Player Profile Dashboard")
    st.caption("Profile, market value trend, and on-pitch activity in one view.")

    config = get_data_source_config()
    show_data_source_help(config)

    if config["mode"] == "bigquery":
        try:
            player_options = load_player_options_bigquery(config)
        except Exception as exc:
            st.error("Unable to load player options from BigQuery.")
            st.exception(exc)
            return
    else:
        try:
            profile_df = load_table("player_profile", config)
            value_history_df = load_table("market_value_history", config)
            heatmap_df = load_table("player_heatmap", config)
        except Exception as exc:
            st.error("Unable to load player profile data.")
            st.exception(exc)
            st.markdown(
                """
                To develop locally first, prepare these parquet files:
                - `artifacts/player_profile/player_profile.parquet`
                - `artifacts/player_profile/market_value_history.parquet`
                - `artifacts/player_profile/player_heatmap.parquet`

                Or configure GCS / BigQuery access with the sidebar variables.
                """
            )
            return

    if config["mode"] != "bigquery":
        validate_required_columns(
            profile_df,
            [
                "player_key",
                "player_name",
                "position",
                "age",
                "height",
                "preferred_foot",
                "birth_country",
                "citizenship",
                "current_club",
                "career_stage",
            ],
            REQUIRED_TABLES["player_profile"],
        )
        validate_required_columns(
            value_history_df,
            [
                "player_key",
                "date",
                "market_value",
                "club_name",
                "age",
            ],
            REQUIRED_TABLES["market_value_history"],
        )
        validate_required_columns(
            heatmap_df,
            [
                "player_key",
                "event_type_group",
                "grid_x",
                "grid_y",
                "event_count",
                "event_weight",
            ],
            REQUIRED_TABLES["player_heatmap"],
        )

        profile_df = profile_df.copy()
        value_history_df = value_history_df.copy()
        heatmap_df = heatmap_df.copy()

        profile_df["player_key"] = profile_df["player_key"].astype(str)
        value_history_df["player_key"] = value_history_df["player_key"].astype(str)
        heatmap_df["player_key"] = heatmap_df["player_key"].astype(str)

        player_options = (
            profile_df[["player_key", "player_name", "current_club"]]
            .drop_duplicates()
            .sort_values(["player_name", "current_club"])
        )

    player_options = player_options.copy()
    player_options["player_key"] = player_options["player_key"].astype(str)
    player_options["label"] = player_options["player_name"] + " | " + player_options["current_club"].fillna("Unknown Club")

    selected_label = st.selectbox("Select Player", player_options["label"].tolist())
    selected_player = player_options[player_options["label"] == selected_label].iloc[0]
    selected_key = selected_player["player_key"]

    if config["mode"] == "bigquery":
        try:
            profile_df, player_value_history, player_heatmap = load_bigquery_player_bundle(selected_key, config)
        except Exception as exc:
            st.error("Unable to load selected player data from BigQuery.")
            st.exception(exc)
            return

        validate_required_columns(
            profile_df,
            [
                "player_key",
                "player_name",
                "position",
                "age",
                "height",
                "preferred_foot",
                "birth_country",
                "citizenship",
                "current_club",
                "career_stage",
            ],
            REQUIRED_TABLES["player_profile"],
        )
        validate_required_columns(
            player_value_history,
            [
                "player_key",
                "date",
                "market_value",
                "club_name",
                "age",
            ],
            REQUIRED_TABLES["market_value_history"],
        )
        validate_required_columns(
            player_heatmap,
            [
                "player_key",
                "event_type_group",
                "grid_x",
                "grid_y",
                "event_count",
                "event_weight",
            ],
            REQUIRED_TABLES["player_heatmap"],
        )

        profile_df = profile_df.copy()
        player_value_history = player_value_history.copy()
        player_heatmap = player_heatmap.copy()
        profile_df["player_key"] = profile_df["player_key"].astype(str)
        player_value_history["player_key"] = player_value_history["player_key"].astype(str)
        player_heatmap["player_key"] = player_heatmap["player_key"].astype(str)
        profile_row = profile_df.iloc[0]
    else:
        profile_row = profile_df[profile_df["player_key"] == selected_key].iloc[0]
        player_value_history = value_history_df[value_history_df["player_key"] == selected_key].copy()
        player_heatmap = heatmap_df[heatmap_df["player_key"] == selected_key].copy()

    if player_value_history.empty:
        st.warning("No market value history found for this player.")
        return

    if player_heatmap.empty:
        st.warning("No heatmap data found for this player.")
        return

    render_profile_cards(profile_row)
    render_value_summary(get_latest_summary_row(player_value_history))
    render_market_value_chart(player_value_history)
    render_heatmaps(player_heatmap, profile_row)


if __name__ == "__main__":
    main()
