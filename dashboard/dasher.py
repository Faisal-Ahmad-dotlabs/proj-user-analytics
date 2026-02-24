import json
import logging
from datetime import timedelta, datetime
from typing import Any

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
from dash import dcc, html, Input, Output, State, callback_context, dash_table

from . import config, ella, mixpanel

logger = logging.getLogger(__name__)
settings = config.get_app_settings()

ALL_USERS_VALUE = "__ALL__"

ENV_OPTIONS = [
    {"label": "Dev", "value": "dev"},
    {"label": "Prod", "value": "prod"},
]

# -----------------------------
# GLOBALS
# -----------------------------

dash_kwargs: dict[str, Any] = {
    "external_stylesheets": [dbc.themes.BOOTSTRAP],
}

if settings.PROXY_PATH:
    dash_kwargs |= {
        "requests_pathname_prefix": f"/{settings.PROXY_PATH}/",
    }

app = dash.Dash(__name__, **dash_kwargs)

app.title = "Ella User Analytics Dashboard"
app.config.suppress_callback_exceptions = True

df = pd.DataFrame()

# -----------------------------
# HELPERS
# -----------------------------
def find_column(df_, patterns):
    """Return first matching column name from patterns (exact match)."""
    for p in patterns:
        if p in df_.columns:
            return p
    return None

def _pick_col(df_, candidates):
    return find_column(df_, candidates)

def get_content_id_col(df_):
    return find_column(df_, [
        "properties.content_id",  # API format (most common)
        "content_id",             # Mixpanel CSV export format
        "properties.Content ID",
        "properties.contentId",
        "properties.ContentId",
    ])

def get_playback_percentage_col(df_):
    return find_column(df_, [
        "properties.playback.playback_percentage",  # API format (flattened)
        "playback_percentage",                      # CSV parsed
        "properties.playback_percentage",           # legacy flattened
        "properties.Playback.Playback percentage",  # old CSV headers
        "properties.Playback.Playback_percentage",
    ])

def get_playback_time_col(df_):
    return find_column(df_, [
        "properties.playback.playback_time",        # API format (flattened)
        "playback_time",                            # CSV parsed
        "properties.playback_time",                 # legacy flattened
        "properties.Playback.Playback time",        # old CSV headers
        "properties.Playback.Playback_time",
    ])

def get_session_id_col(df_):
    return find_column(df_, [
        "properties.session_id",     # API format
        "session_id",                # CSV export format
        "properties.Session ID",     # old CSV headers
        "properties.SessionID",
    ])

def get_distinct_id_col(df_):
    return find_column(df_, [
        "properties.distinct_id",  # API format
        "Distinct ID",             # Mixpanel CSV export format
        "distinct_id",             # alternative CSV
    ])

def get_action_col(df_):
    return find_column(df_, ["properties.action", "action", "properties.Action"])

def get_user_id_col(df_):
    return find_column(df_, [
        "properties.$user_id",   # Mixpanel identified user
        "properties.user_id",    # alternative
        "$user_id",              # CSV export
        "user_id",               # generic
    ])

def parse_time_to_seconds(time_str):
    """Convert time string (MM:SS or HH:MM:SS) or numeric seconds to seconds."""
    if pd.isna(time_str) or time_str == "":
        return 0
    try:
        if isinstance(time_str, (int, float)):
            return float(time_str)
        s = str(time_str).strip()
        parts = s.split(":")
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + int(sec)
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + int(sec)
        return 0
    except Exception:
        return 0

def parse_playback_json(df_):
    """Parse nested playback JSON in CSV exports into flat columns.

    Mixpanel Export API data is already flattened by pd.json_normalize, so we leave it.
    """
    if len(df_) == 0:
        return df_

    # If already flattened (Export API)
    if "properties.playback.playback_percentage" in df_.columns or "properties.playback.playback_time" in df_.columns:
        return df_

    playback_col = find_column(df_, ["playback", "properties.playback"])
    if playback_col is None:
        return df_

    def extract(val, key):
        if pd.isna(val) or val == "":
            return None
        try:
            if isinstance(val, str):
                data = json.loads(val)
            elif isinstance(val, dict):
                data = val
            else:
                return None
            return data.get(key)
        except Exception:
            return None

    if "playback_percentage" not in df_.columns:
        df_["playback_percentage"] = df_[playback_col].apply(lambda v: extract(v, "playback_percentage"))
    if "playback_time" not in df_.columns:
        df_["playback_time"] = df_[playback_col].apply(lambda v: extract(v, "playback_time"))

    return df_

def normalize_df_inplace(df_):
    """Normalize Mixpanel Export API + Mixpanel CSV exports into a consistent schema."""
    if len(df_) == 0:
        return df_

    # Mixpanel CSV export normalization (keep original cols but add API-like cols when missing)
    csv_to_api_mapping = {
        "Event Name": "event",
        "Time": "properties.time",
        "Distinct ID": "properties.distinct_id",
        "City": "properties.$city",
        "Operating System": "properties.$os",
        "Country": "properties.$country",
    }
    for csv_name, api_name in csv_to_api_mapping.items():
        if csv_name in df_.columns and api_name not in df_.columns:
            df_[api_name] = df_[csv_name]

    # datetime + date
    time_col = find_column(df_, ["properties.time", "Time", "time"])
    if time_col:
        df_["datetime"] = pd.to_datetime(df_[time_col], unit="s", errors="coerce")
    else:
        df_["datetime"] = pd.NaT

    df_["date"] = pd.to_datetime(df_["datetime"].dt.date, errors="coerce")

    # Playback JSON parsing for CSV exports
    df_ = parse_playback_json(df_)

    return df_

# -----------------------------
# SETUP
# -----------------------------
def setup() -> None:
    logger.info("dasher: setup")

    try:
        df = pd.read_csv("mixpanel_data.csv", low_memory=False)
        df = normalize_df_inplace(df)
        logger.info(f"dasher: loaded {len(df):,} events from CSV")
    except Exception:
        df = pd.DataFrame()
        print("dasher: during loading No CSV found")


def update_user_dropdown_options():
    """Update dropdown with distinct_id and user_id (when available).

    The dropdown *value* is always the distinct_id so existing filtering logic
    is unchanged. The *label* includes both IDs so Dash's built-in text search
    works for either identifier.
    """
    if len(df) == 0:
        return [{"label": "All users", "value": ALL_USERS_VALUE}]

    distinct_col = _pick_col(df, ["properties.distinct_id", "distinct_id"])
    if not distinct_col:
        return [{"label": "All users", "value": ALL_USERS_VALUE}]

    uid_col = get_user_id_col(df)

    # Build a mapping: distinct_id -> most-common non-null user_id (if any)
    uid_map: dict[str, str] = {}
    if uid_col:
        for did, group in df.groupby(distinct_col):
            uids = group[uid_col].dropna().astype(str)
            uids = uids[uids != ""]
            if len(uids) > 0:
                uid_map[str(did)] = str(uids.mode().iloc[0])

    unique_users = sorted(df[distinct_col].dropna().unique().tolist())
    opts = [{"label": "All users", "value": ALL_USERS_VALUE}]

    for did in unique_users:
        did_str = str(did)
        uid_str = uid_map.get(did_str)

        if uid_str:
            # Show user_id prominently; include distinct_id for search/reference
            did_short = (did_str[:40] + "…") if len(did_str) > 40 else did_str
            uid_short = (uid_str[:40] + "…") if len(uid_str) > 40 else uid_str
            label = f"{uid_short}  [{did_short}]"
        else:
            label = (did_str[:80] + "…") if len(did_str) > 80 else did_str

        opts.append({"label": label, "value": did_str})

    return opts


user_dropdown_options = update_user_dropdown_options()

min_date_data = (
    df["date"].min()
    if len(df) > 0 and "date" in df.columns
    else datetime.now().date()
)

max_date_data = (
    df["date"].max()
    if len(df) > 0 and "date" in df.columns
    else datetime.now().date()
)


# -----------------------------
# METRICS
# -----------------------------
def count_saved_words_by_content(scope_df):
    cid_col = get_content_id_col(scope_df)
    if cid_col is None:
        return {}

    sw = scope_df[scope_df["event"] == "saved_words"].copy()
    if len(sw) == 0:
        return {}

    action_col = get_action_col(sw)
    if action_col is not None:
        sw[action_col] = sw[action_col].astype(str)
        sw = sw[sw[action_col].str.lower() == "add"]

    word_col = _pick_col(sw, ["properties.word", "word", "properties.Word"])
    if word_col is None:
        counts = sw.groupby(cid_col).size().to_dict()
        return {k: int(v) for k, v in counts.items()}

    sw[word_col] = sw[word_col].astype(str)
    counts = sw.groupby(cid_col)[word_col].nunique().to_dict()
    return {k: int(v) for k, v in counts.items()}

def list_saved_words_for_content(scope_df, content_id):
    cid_col = get_content_id_col(scope_df)
    if cid_col is None or content_id is None:
        return []

    sw = scope_df[(scope_df["event"] == "saved_words") & (scope_df[cid_col] == content_id)].copy()
    if len(sw) == 0:
        return []

    action_col = get_action_col(sw)
    if action_col is not None:
        sw[action_col] = sw[action_col].astype(str)
        sw = sw[sw[action_col].str.lower() == "add"]

    word_col = _pick_col(sw, ["properties.word", "word", "properties.Word"])
    if word_col is None:
        return []

    words = sorted(set(sw[word_col].dropna().astype(str).tolist()), key=lambda x: x.lower())
    return words

def get_content_table_data(env_value: str, scope_df):
    """Per content: max playback_percentage and max playback_time (from playback events only)."""

    cid_col = get_content_id_col(scope_df)
    pct_col = get_playback_percentage_col(scope_df)
    time_col = get_playback_time_col(scope_df)

    if cid_col is None or "event" not in scope_df.columns:
        return []

    playback_events = scope_df[scope_df["event"] == "playback"].copy()
    if len(playback_events) == 0:
        return []

    playback_events = playback_events[playback_events[cid_col].notna()].copy()
    if len(playback_events) == 0:
        return []

    content_ids = playback_events[cid_col].dropna().unique().tolist()
    if not content_ids:
        return []

    content_details = ella.fetch_multiple_contents(env_value, content_ids)
    saved_counts = count_saved_words_by_content(scope_df)

    rows = []
    for content_id in content_ids:
        content_playback = playback_events[playback_events[cid_col] == content_id]

        max_progress = 0.0
        if pct_col and pct_col in content_playback.columns:
            vals = pd.to_numeric(content_playback[pct_col], errors="coerce").dropna()
            if len(vals) > 0:
                max_progress = float(vals.max())

        max_time_seconds = 0
        if time_col and time_col in content_playback.columns:
            tvals = content_playback[time_col].dropna()
            if len(tvals) > 0:
                max_time_seconds = max([parse_time_to_seconds(t) for t in tvals])

        details = content_details.get(content_id, {})
        upload_date = "-"
        if details.get("upload_date"):
            try:
                upload_date = pd.to_datetime(details["upload_date"]).strftime("%Y-%m-%d")
            except Exception:
                upload_date = "-"

        listen_url = f"https://helloella-prod.web.app/listen/{content_id}"

        rows.append({
            "content_id": content_id,
            "title": details.get("title", f"Content {str(content_id)[:8]}..."),
            "type": details.get("type", "-"),
            "upload_date": upload_date,
            "words": int(details.get("word_count", 0) or 0),
            "progress": round(max_progress, 1),
            "playback_time": round(max_time_seconds / 60, 1),
            "saved_words": int(saved_counts.get(content_id, 0)),
            "listen_link": f"[Open]({listen_url})",
        })

    rows.sort(key=lambda x: x["progress"], reverse=True)
    return rows


def calculate_content_metrics(env_value: str, scope_df):
    """Aggregate content metrics using playback events only.

    - total_contents: unique content_ids that have at least 1 playback event
    - avg_progress: average of per-content max playback_percentage
    - avg_playback_time: average of per-content max playback_time (minutes)
    - total_words_completed: words * (max_progress/100) summed per content
    """

    default_metrics = {
        "total_contents": 0,
        "total_words_uploaded": 0,
        "total_words_completed": 0,
        "avg_progress": 0,
        "avg_playback_time": 0,
    }

    if len(scope_df) == 0 or "event" not in scope_df.columns:
        return default_metrics

    cid_col = get_content_id_col(scope_df)
    if cid_col is None:
        return default_metrics

    playback_events = scope_df[scope_df["event"] == "playback"].copy()
    if len(playback_events) == 0:
        return default_metrics

    playback_events = playback_events[playback_events[cid_col].notna()].copy()
    if len(playback_events) == 0:
        return default_metrics

    content_ids = playback_events[cid_col].dropna().unique().tolist()
    total_contents = int(len(content_ids))

    content_details = ella.fetch_multiple_contents(env_value, content_ids)
    total_words_uploaded = int(sum([d.get("word_count", 0) or 0 for d in content_details.values()]))

    pct_col = get_playback_percentage_col(playback_events)
    time_col = get_playback_time_col(playback_events)

    # Progress
    max_progress_per_content = {}
    if pct_col and pct_col in playback_events.columns:
        playback_events[pct_col] = pd.to_numeric(playback_events[pct_col], errors="coerce")
        for cid in content_ids:
            vals = playback_events.loc[playback_events[cid_col] == cid, pct_col].dropna()
            if len(vals) > 0:
                max_progress_per_content[cid] = float(vals.max())

    avg_progress = 0.0
    total_words_completed = 0
    if max_progress_per_content:
        avg_progress = float(sum(max_progress_per_content.values()) / len(max_progress_per_content))
        for cid, max_prog in max_progress_per_content.items():
            wc = content_details.get(cid, {}).get("word_count", 0) or 0
            total_words_completed += int(wc * (max_prog / 100.0))

    # Playback time
    max_time_per_content = {}
    if time_col and time_col in playback_events.columns:
        for cid in content_ids:
            tvals = playback_events.loc[playback_events[cid_col] == cid, time_col].dropna()
            if len(tvals) > 0:
                max_secs = max([parse_time_to_seconds(t) for t in tvals])
                if max_secs > 0:
                    max_time_per_content[cid] = max_secs

    avg_playback_time = 0.0
    if max_time_per_content:
        avg_playback_time = float(sum(max_time_per_content.values()) / len(max_time_per_content) / 60.0)

    return {
        "total_contents": total_contents,
        "total_words_uploaded": total_words_uploaded,
        "total_words_completed": int(total_words_completed),
        "avg_progress": round(avg_progress, 1) if avg_progress > 0 else 0,
        "avg_playback_time": round(avg_playback_time, 1) if avg_playback_time > 0 else 0,
    }


def create_session_timeline(scope_df, min_date, max_date, is_all_users=False):
    """Create sessions-per-day timeline with correct session counting across schema variants."""

    if len(scope_df) == 0 or "date" not in scope_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No session data available", height=400, plot_bgcolor="white", paper_bgcolor="white")
        return fig, 0, 0

    filtered_df = scope_df[(scope_df["date"] >= min_date) & (scope_df["date"] <= max_date)].copy()

    session_col = get_session_id_col(filtered_df)
    if session_col is None:
        fig = go.Figure()
        fig.update_layout(title="No session_id column found", height=400, plot_bgcolor="white", paper_bgcolor="white")
        return fig, 0, 0

    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")
    rows = []

    for date in all_dates:
        date_ts = pd.Timestamp(date.date())
        day_events = filtered_df[filtered_df["date"] == date_ts]

        if len(day_events) == 0:
            rows.append({"date": date_ts, "sessions": 0, "total_time": "0 min", "avg_duration": "0 min"})
            continue

        sessions_today = day_events[session_col].dropna().unique()
        num_sessions = int(len(sessions_today))

        if num_sessions == 0:
            rows.append({"date": date_ts, "sessions": 0, "total_time": "0 min", "avg_duration": "0 min"})
            continue

        session_durations = []
        for sid in sessions_today:
            s_events = day_events[day_events[session_col] == sid]
            if len(s_events) == 0:
                continue
            start = s_events["datetime"].min()
            end = s_events["datetime"].max()
            if pd.isna(start) or pd.isna(end):
                continue
            session_durations.append((end - start).total_seconds() / 60)

        total_time = float(sum(session_durations))
        avg_time = float(total_time / num_sessions) if num_sessions > 0 else 0.0

        rows.append({
            "date": date_ts,
            "sessions": num_sessions,
            "total_time": f"{int(total_time)} min",
            "avg_duration": f"{int(avg_time)} min",
        })

    timeline_df = pd.DataFrame(rows)
    days_shown = len(timeline_df)
    total_days = int((max_date - min_date).days + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline_df["date"],
        y=timeline_df["sessions"],
        mode="lines+markers",
        line=dict(color="#7c3aed", width=2),
        marker=dict(size=8, color="#7c3aed"),
        customdata=timeline_df[["total_time", "avg_duration"]],
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            "Sessions: %{y}<br>"
            "Total time: %{customdata[0]}<br>"
            "Avg duration: %{customdata[1]}<extra></extra>"
        ),
        name="Sessions",
    ))

    title_text = "All Users - Sessions Per Day" if is_all_users else None

    fig.update_layout(
        title=title_text,
        height=400,
        xaxis_title="",
        yaxis_title="Sessions",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickformat="%b %d", dtick="D1" if days_shown <= 10 else None),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", rangemode="tozero", tick0=0, dtick=1),
        margin=dict(l=60, r=20, t=45, b=50),
        hovermode="closest",
        showlegend=False,
    )

    return fig, days_shown, total_days


def get_session_details_with_content(env_value: str, scope_df, date_ts):
    """Session details for a day. Content metrics computed from playback events only."""

    if len(scope_df) == 0 or "date" not in scope_df.columns:
        return []

    day_data = scope_df[scope_df["date"] == date_ts]
    if len(day_data) == 0:
        return []

    session_col = get_session_id_col(day_data)
    cid_col = get_content_id_col(day_data)

    if session_col is None:
        return []

    pct_col = get_playback_percentage_col(day_data)
    time_col = get_playback_time_col(day_data)

    sessions = []
    for session_id, session_data in day_data.groupby(session_col):
        session_data = session_data.sort_values("datetime")
        start_time = session_data["datetime"].min()
        end_time = session_data["datetime"].max()
        duration_mins = int((end_time - start_time).total_seconds() / 60) if pd.notna(start_time) and pd.notna(end_time) else 0

        event_counts = session_data["event"].value_counts().to_dict() if "event" in session_data.columns else {}
        top_events = session_data["event"].value_counts().head(5).to_dict() if "event" in session_data.columns else {}

        content_played = []
        if cid_col is not None and "event" in session_data.columns:
            session_playback = session_data[(session_data["event"] == "playback") & (session_data[cid_col].notna())]
            if len(session_playback) > 0:
                content_ids = session_playback[cid_col].unique().tolist()
                content_details = ella.fetch_multiple_contents(env_value, content_ids)

                for cid in content_ids:
                    cid_playback = session_playback[session_playback[cid_col] == cid].sort_values("datetime")

                    old_progress = 0.0
                    new_progress = 0.0
                    if pct_col and pct_col in cid_playback.columns:
                        prog_vals = pd.to_numeric(cid_playback[pct_col], errors="coerce").dropna()
                        if len(prog_vals) > 0:
                            old_progress = float(prog_vals.iloc[0])
                            new_progress = float(prog_vals.iloc[-1])

                    progress_change = new_progress - old_progress

                    max_time_seconds = 0
                    if time_col and time_col in cid_playback.columns:
                        tvals = cid_playback[time_col].dropna()
                        if len(tvals) > 0:
                            max_time_seconds = max([parse_time_to_seconds(t) for t in tvals])

                    details = content_details.get(cid, {})
                    content_played.append({
                        "content_id": cid,
                        "title": details.get("title", f"Content {str(cid)[:8]}..."),
                        "old_progress": round(old_progress, 1),
                        "new_progress": round(new_progress, 1),
                        "progress_change": round(progress_change, 1),
                        "playback_time_min": round(max_time_seconds / 60, 1) if max_time_seconds > 0 else 0.0,
                    })

        sessions.append({
            "session_id": session_id,
            "start_time": start_time.strftime("%H:%M") if pd.notna(start_time) else "N/A",
            "end_time": end_time.strftime("%H:%M") if pd.notna(end_time) else "N/A",
            "duration": duration_mins,
            "event_count": int(len(session_data)),
            "event_types": int(len(event_counts)),
            "top_events": top_events,
            "content_played": content_played,
        })

    sessions.sort(key=lambda x: x["start_time"])
    return sessions


def get_content_history(scope_df, content_id, date_filter="month"):
    """Daily history for a content based on playback events only.

    Returns:
      history_df: date, max_progress, playback_time_minutes
      summary: overall_max_progress, total_playback_time_minutes, days_interacted, total_days
    """

    cid_col = get_content_id_col(scope_df)
    if cid_col is None or "event" not in scope_df.columns or "date" not in scope_df.columns:
        return pd.DataFrame(), {}

    content_playback = scope_df[(scope_df["event"] == "playback") & (scope_df[cid_col] == content_id)].copy()
    if len(content_playback) == 0:
        return pd.DataFrame(), {}

    pct_col = get_playback_percentage_col(content_playback)
    time_col = get_playback_time_col(content_playback)

    max_date = content_playback["date"].max()
    if pd.isna(max_date):
        return pd.DataFrame(), {}

    if date_filter == "week":
        min_date = max_date - timedelta(days=6)
    else:
        min_date = max_date - timedelta(days=29)

    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")
    daily_rows = []

    for date in all_dates:
        date_ts = pd.Timestamp(date.date())
        day_playback = content_playback[content_playback["date"] == date_ts]

        max_progress = 0.0
        if len(day_playback) > 0 and pct_col and pct_col in day_playback.columns:
            vals = pd.to_numeric(day_playback[pct_col], errors="coerce").dropna()
            if len(vals) > 0:
                max_progress = float(vals.max())

        playback_time_seconds = 0
        if len(day_playback) > 0 and time_col and time_col in day_playback.columns:
            tvals = day_playback[time_col].dropna()
            if len(tvals) > 0:
                playback_time_seconds = max([parse_time_to_seconds(t) for t in tvals])

        daily_rows.append({
            "date": date_ts,
            "max_progress": max_progress,
            "playback_time_minutes": playback_time_seconds / 60.0,
        })

    history_df = pd.DataFrame(daily_rows).sort_values("date")

    all_progress_vals = history_df.loc[history_df["max_progress"] > 0, "max_progress"].tolist()
    all_time_vals = history_df.loc[history_df["playback_time_minutes"] > 0, "playback_time_minutes"].tolist()

    summary = {
        "overall_max_progress": round(max(all_progress_vals), 1) if all_progress_vals else 0.0,
        "total_playback_time_minutes": round(sum(all_time_vals), 1) if all_time_vals else 0.0,
        "days_interacted": int(len(history_df[(history_df["max_progress"] > 0) | (history_df["playback_time_minutes"] > 0)])),
        "total_days": int(len(all_dates)),
    }

    return history_df, summary


def get_user_languages(user_df) -> tuple[str, str]:
    selector_events = user_df[
        (user_df["event"] == "selector") &
        (user_df.get("properties.Page Name", user_df.get("properties.page_name", "")) == "Onboarding Language")
    ].copy()

    if len(selector_events) == 0:
        return "Unknown", "Unknown"

    target_language = "Unknown"
    native_language = "Unknown"

    selector_name_col = _pick_col(selector_events, ["properties.Selector Name", "properties.selector_name", "Selector Name"])
    selector_option_col = _pick_col(selector_events, ["properties.Selected Option", "properties.selected_option", "Selected Option"])

    if selector_name_col and selector_option_col:
        for _, row in selector_events.iterrows():
            selector_name = str(row.get(selector_name_col, "")).lower()
            selector_option = str(row.get(selector_option_col, "Unknown"))
            if "foreign" in selector_name or "target" in selector_name:
                target_language = selector_option
            elif "native" in selector_name:
                native_language = selector_option

    return target_language, native_language

# -----------------------------
# MOBILE-FRIENDLY STYLES
# -----------------------------
mobile_styles = """
<style>
/* Base responsive adjustments */
@media (max-width: 768px) {
    h1 { font-size: 24px !important; }

    .card-mobile { padding: 12px !important; margin-bottom: 10px !important; }

    .dash-table-container { overflow-x: auto !important; -webkit-overflow-scrolling: touch !important; }

    button { min-height: 44px !important; font-size: 14px !important; }

    input, .Select-control { min-height: 44px !important; font-size: 16px !important; }

    .Select-menu-outer { max-height: 300px !important; }

    .js-plotly-plot { width: 100% !important; }

    .row { margin-left: 0 !important; margin-right: 0 !important; }
}

@media (max-width: 480px) {
    .card-mobile { padding: 8px !important; }
    body { font-size: 14px !important; }
}

@media (hover: none) and (pointer: coarse) {
    button, a, .dash-dropdown { min-height: 44px !important; min-width: 44px !important; }
}
</style>
"""

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        ''' + mobile_styles + '''
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def card_style():
    return {
        "backgroundColor": "white",
        "padding": "20px",
        "borderRadius": "14px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
        "marginBottom": "14px"
    }

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1("Dashboard", style={"fontSize": "32px", "fontWeight": "700", "color": "#111827", "marginBottom": "4px"}),
            html.P("User analytics and insights", style={"color": "#6b7280", "fontSize": "14px", "marginBottom": "0"}),
        ], style={"maxWidth": "1200px", "margin": "0 auto"}),
    ], style={"padding": "26px 18px", "backgroundColor": "#f5f5f5"}),

    dbc.Toast(
        id="toast",
        header="",
        is_open=False,
        dismissable=True,
        duration=6000,
        icon="primary",
        style={
            "position": "fixed",
            "top": "18px",
            "right": "18px",
            "width": "min(420px, calc(100vw - 36px))",
            "zIndex": 9999,
            "maxWidth": "100%"
        },
    ),

    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div("Fetch Mixpanel Data", style={"fontSize": "16px", "fontWeight": "700", "color": "#111827"}),
                    html.Div("Choose env and date range", style={"fontSize": "12px", "color": "#6b7280"}),
                ]),
                dbc.Badge(id="fetch-badge", children="Idle", color="secondary", pill=True),
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "14px"}),

            dbc.Row([
                dbc.Col([
                    html.Label("Data source", style={"fontWeight": "700", "fontSize": "12px", "color": "#374151"}),
                    dcc.Dropdown(
                        id="env-dropdown",
                        options=ENV_OPTIONS,
                        value="dev",
                        clearable=False,
                    ),
                ], xs=12, sm=6, md=3, lg=3),
                dbc.Col([
                    html.Label("From date", style={"fontWeight": "700", "fontSize": "12px", "color": "#374151"}),
                    dcc.DatePickerSingle(
                        id="mixpanel-from-date",
                        date=datetime.now().date() - timedelta(days=30),
                        display_format="YYYY-MM-DD",
                        style={"width": "100%"},
                    ),
                ], xs=12, sm=6, md=3, lg=3),
                dbc.Col([
                    html.Label("To date", style={"fontWeight": "700", "fontSize": "12px", "color": "#374151"}),
                    dcc.DatePickerSingle(
                        id="mixpanel-to-date",
                        date=datetime.now().date(),
                        display_format="YYYY-MM-DD",
                        style={"width": "100%"},
                    ),
                ], xs=12, sm=6, md=3, lg=3),
                dbc.Col([
                    html.Label(" ", style={"display": "block"}),
                    dbc.Button("Fetch data", id="fetch-mixpanel-btn", color="success", style={"width": "100%", "fontWeight": "700"}),
                ], xs=12, sm=6, md=3, lg=3),
            ], className="g-2"),

            html.Div(id="mixpanel-fetch-status", style={"marginTop": "12px", "fontSize": "13px", "color": "#374151"}),

            dcc.Loading(
                id="fetch-loading",
                type="default",
                children=html.Div(id="fetch-loading-dummy"),
                style={"marginTop": "8px"},
            ),
        ], style=card_style()),

        html.Div(style={"height": 14}),

        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Select user (Distinct ID / User ID)", style={"fontWeight": "700", "fontSize": "12px", "color": "#374151"}),
                    dcc.Dropdown(
                        id="user-dropdown",
                        options=user_dropdown_options,
                        value=ALL_USERS_VALUE,
                        searchable=True,
                        clearable=False,
                    ),
                ], width=12),
            ]),

            html.Div(id="selected-user-id-display", style={"marginTop": "8px"}),

            html.Div(style={"height": 14}),

            html.Div([
                html.Span("Total unique users: ", style={"color": "#6b7280", "fontSize": "13px", "fontWeight": "700"}),
                html.Span(id="total-unique-users", style={"color": "#111827", "fontSize": "13px", "fontWeight": "900"}),
            ]),
        ], style=card_style()),

        html.Div(style={"height": 14}),

        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("User overview", style={"fontSize": "16px", "fontWeight": "800", "color": "#111827", "marginBottom": "14px"}),
                    html.Div([html.Span("Sign-up date: ", style={"color": "#6b7280"}), html.Span(id="signup-date", style={"fontWeight": "800"})], style={"marginBottom": "10px"}),
                    html.Div([html.Span("Total sessions: ", style={"color": "#6b7280"}), html.Span(id="total-sessions", style={"fontWeight": "800"})], style={"marginBottom": "10px"}),
                    html.Div([html.Span("Active days: ", style={"color": "#6b7280"}), html.Span(id="active-days", style={"fontWeight": "800"})], style={"marginBottom": "10px"}),
                    html.Div([html.Span("Last seen: ", style={"color": "#6b7280"}), html.Span(id="last-seen", style={"fontWeight": "800"})], style={"marginBottom": "10px"}),
                    html.Div([html.Span("Target language: ", style={"color": "#6b7280"}), html.Span(id="target-language", style={"fontWeight": "800"})], style={"marginBottom": "10px"}),
                    html.Div([html.Span("Native language: ", style={"color": "#6b7280"}), html.Span(id="native-language", style={"fontWeight": "800"})]),
                ], style=card_style()),
            ], xs=12, sm=12, md=6, lg=6),

            dbc.Col([
                html.Div([
                    html.Div("User attributes", style={"fontSize": "16px", "fontWeight": "800", "color": "#111827", "marginBottom": "14px"}),
                    html.Div([html.Span("Location: ", style={"color": "#6b7280"}), html.Span(id="user-location", style={"fontWeight": "800"})], style={"marginBottom": "10px"}),
                    html.Div([html.Span("Device: ", style={"color": "#6b7280"}), html.Span(id="user-device", style={"fontWeight": "800"})], style={"marginBottom": "10px"}),
                    html.Div([html.Span("Top event: ", style={"color": "#6b7280"}), html.Span(id="top-event", style={"fontWeight": "800"})], style={"marginBottom": "10px"}),
                    html.Div([html.Span("Total events: ", style={"color": "#6b7280"}), html.Span(id="total-events", style={"fontWeight": "800"})]),
                ], style=card_style()),
            ], xs=12, sm=12, md=6, lg=6),
        ], id="user-overview-row"),

        html.Div(style={"height": 14}),

        html.Div([
            html.Div("Content library overview", style={"fontSize": "16px", "fontWeight": "800", "color": "#111827", "marginBottom": "14px"}),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Total contents", style={"fontSize": "12px", "color": "#2563eb", "fontWeight": "800"}),
                        html.Div(id="metric-total-contents", style={"fontSize": "28px", "fontWeight": "900", "color": "#111827"}),
                    ], style={"backgroundColor": "#e6f2ff", "padding": "16px", "borderRadius": "12px"}),
                ], xs=12, sm=6, md=4, lg=4),
                dbc.Col([
                    html.Div([
                        html.Div("Total words uploaded", style={"fontSize": "12px", "color": "#16a34a", "fontWeight": "800"}),
                        html.Div(id="metric-words-uploaded", style={"fontSize": "28px", "fontWeight": "900", "color": "#111827"}),
                    ], style={"backgroundColor": "#d1fae5", "padding": "16px", "borderRadius": "12px"}),
                ], xs=12, sm=6, md=4, lg=4),
                dbc.Col([
                    html.Div([
                        html.Div("Total words completed", style={"fontSize": "12px", "color": "#7c3aed", "fontWeight": "800"}),
                        html.Div(id="metric-words-completed", style={"fontSize": "28px", "fontWeight": "900", "color": "#111827"}),
                    ], style={"backgroundColor": "#f3e8ff", "padding": "16px", "borderRadius": "12px"}),
                ], xs=12, sm=6, md=4, lg=4),
            ], className="g-2", style={"marginBottom": "10px"}),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Avg progress", style={"fontSize": "12px", "color": "#f59e0b", "fontWeight": "800"}),
                        html.Div(id="metric-avg-progress", style={"fontSize": "28px", "fontWeight": "900", "color": "#111827"}),
                    ], style={"backgroundColor": "#fef3c7", "padding": "16px", "borderRadius": "12px"}),
                ], xs=12, sm=6, md=6, lg=6),
                dbc.Col([
                    html.Div([
                        html.Div("Avg playback time", style={"fontSize": "12px", "color": "#ec4899", "fontWeight": "800"}),
                        html.Div(id="metric-playback-time", style={"fontSize": "28px", "fontWeight": "900", "color": "#111827"}),
                    ], style={"backgroundColor": "#fce7f3", "padding": "16px", "borderRadius": "12px"}),
                ], xs=12, sm=6, md=6, lg=6),
            ], className="g-2"),

            html.Div(style={"height": 14}),
            html.Div([
                html.Span("Date range: ", style={"fontSize": "12px", "color": "#6b7280", "marginRight": "10px", "fontWeight": "800"}),
                dcc.Dropdown(
                    id="content-date-range",
                    options=[
                        {"label": "All time", "value": "all"},
                        {"label": "Last 7 days", "value": "7"},
                        {"label": "Last 30 days", "value": "30"},
                    ],
                    value="all",
                    clearable=False,
                    style={"width": "200px", "display": "inline-block"},
                )
            ]),
            html.Div(id="content-table-container", style={"marginTop": "14px"}),
        ], style=card_style()),

        html.Div(style={"height": 14}),
        html.Div(id="content-details-container"),

        html.Div(style={"height": 14}),

        html.Div([
            html.Div("Session timeline", style={"fontSize": "16px", "fontWeight": "800", "color": "#111827", "marginBottom": "14px"}),

            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id="date-range-dropdown",
                        options=[
                            {"label": "Last 7 days", "value": "7"},
                            {"label": "Last 30 days", "value": "30"},
                            {"label": "Custom range", "value": "custom"},
                        ],
                        value="30",
                        clearable=False,
                        style={"width": "100%", "maxWidth": "240px"},
                    )
                ], xs=12, sm=12, md=4, lg=4),
                dbc.Col([
                    html.Div(id="custom-date-picker-container", style={"display": "none"}, children=[
                        dcc.DatePickerRange(
                            id="custom-date-range",
                            min_date_allowed=min_date_data,
                            max_date_allowed=max_date_data,
                            start_date=max_date_data - timedelta(days=30),
                            end_date=max_date_data,
                            display_format="YYYY-MM-DD",
                        )
                    ]),
                ], xs=12, sm=12, md=8, lg=8),
            ], className="g-2"),

            html.Div(id="date-range-text", style={"marginTop": "8px", "fontSize": "12px", "color": "#6b7280"}),
            dcc.Graph(id="session-timeline", config={"displayModeBar": False}, clear_on_unhover=True),
        ], style=card_style()),

        html.Div(style={"height": 14}),
        html.Div(id="session-details-container"),

    ], style={
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "0 12px 24px 12px",
        "backgroundColor": "#f5f5f5"
    }),

    dcc.Store(id="current-user-store", data=ALL_USERS_VALUE),
    dcc.Store(id="selected-content-store", data=None),
    dcc.Store(id="content-date-filter-store", data="month"),
    dcc.Store(id="env-store", data="dev"),
])

# -----------------------------
# CALLBACKS
# -----------------------------
@app.callback(
    Output("custom-date-picker-container", "style"),
    Input("date-range-dropdown", "value"),
)
def toggle_date_picker(date_range):
    if date_range == "custom":
        return {"display": "block"}
    return {"display": "none"}

@app.callback(
    Output("user-overview-row", "style"),
    Input("user-dropdown", "value"),
)
def toggle_user_overview(user_id):
    if user_id == ALL_USERS_VALUE:
        return {"display": "none"}
    return {"display": "flex"}

@app.callback(
    Output("fetch-badge", "children"),
    Input("fetch-mixpanel-btn", "n_clicks"),
    prevent_initial_call=True,
)
def set_fetch_badge_loading(_):
    return "Loading..."

@app.callback(
    [
        Output("mixpanel-fetch-status", "children"),
        Output("fetch-loading-dummy", "children"),
        Output("user-dropdown", "options"),
        Output("toast", "is_open"),
        Output("toast", "icon"),
        Output("toast", "header"),
        Output("toast", "children"),
        Output("fetch-badge", "children", allow_duplicate=True),
        Output("env-store", "data"),
    ],
    Input("fetch-mixpanel-btn", "n_clicks"),
    [
        State("env-dropdown", "value"),
        State("mixpanel-from-date", "date"),
        State("mixpanel-to-date", "date"),
    ],
    prevent_initial_call=True,
)
def fetch_data_from_mixpanel(n_clicks, env_value, from_date, to_date):
    global df, min_date_data, max_date_data, content_cache

    if not from_date or not to_date:
        msg = dbc.Alert("Select both dates.", color="danger")
        return msg, "", dash.no_update, True, "danger", "Fetch failed", "Missing from/to date.", "Idle", env_value

    from_date_str = datetime.fromisoformat(from_date).strftime("%Y-%m-%d")
    to_date_str = datetime.fromisoformat(to_date).strftime("%Y-%m-%d")

    try:
        new_df = mixpanel.fetch_mixpanel_data(from_date_str, to_date_str, env_value)
    except Exception as err:
        msg = dbc.Alert(f"error: {err}", color="danger")
        return msg, "", dash.no_update, True, "danger", "Fetch failed", err, "Idle", env_value

    df = new_df.copy()
    df = normalize_df_inplace(df)

    content_cache = {}

    df.to_csv("mixpanel_data.csv", index=False)

    if len(df) > 0 and "date" in df.columns:
        min_date_data = df["date"].min()
        max_date_data = df["date"].max()

    new_options = update_user_dropdown_options()

    if len(df) == 0:
        status = dbc.Alert(f"Fetched 0 events ({env_value.upper()}).", color="warning")
        return status, "", new_options, True, "warning", "Fetch complete", "0 events returned.", "Idle", env_value

    status = dbc.Alert(f"Fetched {len(df):,} events ({env_value.upper()}).", color="success")
    return status, "", new_options, True, "success", "Fetch complete", f"Loaded {len(df):,} events.", "Idle", env_value

@app.callback(
    Output("selected-user-id-display", "children"),
    Input("user-dropdown", "value"),
    prevent_initial_call=False,
)
def display_selected_user_id(user_id):
    if not user_id or user_id == ALL_USERS_VALUE:
        return html.Div()

    # Look up the associated user_id for this distinct_id
    uid_str = None
    if len(df) > 0:
        distinct_col = _pick_col(df, ["properties.distinct_id", "distinct_id"])
        uid_col = get_user_id_col(df)
        if distinct_col and uid_col:
            rows = df[df[distinct_col].astype(str) == str(user_id)][uid_col].dropna().astype(str)
            rows = rows[rows != ""]
            if len(rows) > 0:
                uid_str = str(rows.mode().iloc[0])

    id_rows = [
        html.Div([
            html.Span("Distinct ID: ", style={"fontWeight": "700", "color": "#374151", "fontSize": "11px"}),
            html.Code(
                str(user_id),
                style={
                    "backgroundColor": "#f3f4f6",
                    "padding": "4px 8px",
                    "borderRadius": "4px",
                    "fontSize": "11px",
                    "color": "#111827",
                    "userSelect": "all",
                    "cursor": "text",
                    "fontFamily": "monospace",
                    "wordBreak": "break-all",
                }
            ),
        ]),
    ]

    if uid_str:
        id_rows.append(html.Div([
            html.Span("User ID: ", style={"fontWeight": "700", "color": "#374151", "fontSize": "11px"}),
            html.Code(
                uid_str,
                style={
                    "backgroundColor": "#e0f2fe",
                    "padding": "4px 8px",
                    "borderRadius": "4px",
                    "fontSize": "11px",
                    "color": "#0369a1",
                    "userSelect": "all",
                    "cursor": "text",
                    "fontFamily": "monospace",
                    "wordBreak": "break-all",
                }
            ),
        ], style={"marginTop": "6px"}))

    return html.Div([
        dbc.Alert(
            html.Div(id_rows),
            color="light",
            style={"padding": "10px", "marginBottom": "0", "border": "1px solid #e5e7eb"},
        )
    ])

@app.callback(
    Output("total-unique-users", "children"),
    Input("user-dropdown", "options"),
    prevent_initial_call=False,
)
def update_user_counts(options):
    if len(df) == 0:
        return "0"

    col = _pick_col(df, ["properties.distinct_id", "distinct_id"])
    if not col:
        return "0"

    total_users = int(df[col].nunique())
    return f"{total_users:,}"

@app.callback(
    [
        Output("signup-date", "children"),
        Output("total-sessions", "children"),
        Output("active-days", "children"),
        Output("last-seen", "children"),
        Output("target-language", "children"),
        Output("native-language", "children"),
        Output("user-location", "children"),
        Output("user-device", "children"),
        Output("top-event", "children"),
        Output("total-events", "children"),
        Output("metric-total-contents", "children"),
        Output("metric-words-uploaded", "children"),
        Output("metric-words-completed", "children"),
        Output("metric-avg-progress", "children"),
        Output("metric-playback-time", "children"),
        Output("content-table-container", "children"),
        Output("session-timeline", "figure"),
        Output("date-range-text", "children"),
        Output("current-user-store", "data"),
    ],
    [
        Input("user-dropdown", "value"),
        Input("date-range-dropdown", "value"),
        Input("custom-date-range", "start_date"),
        Input("custom-date-range", "end_date"),
        Input("content-date-range", "value"),
        Input("env-store", "data"),
    ],
    prevent_initial_call=False,
)
def update_dashboard(user_id, date_range, custom_start, custom_end, content_date_range, env_value):
    def empty_figure(message):
        fig = go.Figure()
        fig.update_layout(title=message, height=400, plot_bgcolor="white", paper_bgcolor="white")
        return fig

    if user_id is None:
        user_id = ALL_USERS_VALUE
    if not env_value:
        env_value = "dev"

    if len(df) == 0:
        empty_table = html.Div([html.P("No data loaded. Fetch Mixpanel data first.", style={"color": "#6b7280", "textAlign": "center", "padding": "18px"})])
        return (
            "No data",
            "0",
            "0 days",
            "No data",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "N/A",
            "0",
            "0",
            "0",
            "N/A",
            "0%",
            "0 min",
            empty_table,
            empty_figure("No data loaded"),
            "",
            user_id,
        )

    distinct_id_col = _pick_col(df, ["properties.distinct_id", "distinct_id"])

    if user_id == ALL_USERS_VALUE:
        scope_df = df.copy()
        is_all_users = True
    else:
        if distinct_id_col:
            scope_df = df[df[distinct_id_col].astype(str) == str(user_id)].copy()
        else:
            scope_df = df.copy()
        is_all_users = False

    if len(scope_df) == 0:
        empty_table = html.Div([html.P("No data for selected user.", style={"color": "#6b7280", "textAlign": "center", "padding": "18px"})])
        return (
            "No data",
            "0",
            "0 days",
            "No data",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "N/A",
            "0",
            "0",
            "0",
            "N/A",
            "0%",
            "0 min",
            empty_table,
            empty_figure("No data for selection"),
            "",
            user_id,
        )

    first_event = scope_df["datetime"].min()
    last_event = scope_df["datetime"].max()

    first_event_str = first_event.strftime("%b %d, %Y") if pd.notna(first_event) else "N/A"
    last_seen_str = last_event.strftime("%b %d, %Y, %H:%M UTC") if pd.notna(last_event) else "N/A"

    total_sessions = int(scope_df["properties.Session ID"].nunique()) if "properties.Session ID" in scope_df.columns else 0
    active_days = int(scope_df["date"].nunique()) if "date" in scope_df.columns else 0
    total_events = int(len(scope_df))

    target_lang, native_lang = get_user_languages(scope_df)

    user_city = "Unknown"
    if "properties.$city" in scope_df.columns:
        cities = scope_df["properties.$city"].dropna()
        if len(cities) > 0:
            user_city = str(cities.mode().iloc[0])

    user_device = "Unknown"
    if "properties.$os" in scope_df.columns:
        devs = scope_df["properties.$os"].dropna()
        if len(devs) > 0:
            user_device = str(devs.mode().iloc[0])

    top_event = scope_df["event"].value_counts().index[0] if len(scope_df) > 0 else "N/A"

    content_df = scope_df.copy()
    if content_date_range != "all" and "date" in content_df.columns:
        days = int(content_date_range)
        max_date_content = content_df["date"].max()
        min_date_content = max_date_content - timedelta(days=days - 1)
        content_df = content_df[(content_df["date"] >= min_date_content) & (content_df["date"] <= max_date_content)]

    content_metrics = calculate_content_metrics(env_value, content_df)
    table_data = get_content_table_data(env_value, content_df)

    # Content table (always render DataTable so downstream callbacks don't break)
    content_table = html.Div([
        dbc.Alert("No content data available.", color="light", style={"marginBottom": "10px"}) if len(table_data) == 0 else html.Div(),
        dash_table.DataTable(
            id="content-table",
            data=table_data if len(table_data) > 0 else [],
            columns=[
                {"name": "Title", "id": "title"},
                {"name": "Type", "id": "type"},
                {"name": "Upload date", "id": "upload_date"},
                {"name": "Words", "id": "words"},
                {"name": "Progress (%)", "id": "progress", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Playback (min)", "id": "playback_time", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Saved words", "id": "saved_words", "type": "numeric"},
                {"name": "Listen", "id": "listen_link", "presentation": "markdown"},
                {"name": "content_id", "id": "content_id"},
            ],
            hidden_columns=["content_id"],
            markdown_options={"link_target": "_blank"},
            row_selectable=False,
            selected_rows=[],
            active_cell=None,
            style_table={"overflowX": "auto", "minWidth": "100%"},
            style_header={
                "backgroundColor": "#f9fafb",
                "fontWeight": "800",
                "fontSize": "12px",
                "color": "#374151",
                "textAlign": "left",
                "padding": "12px",
                "borderBottom": "1px solid #e5e7eb",
            },
            style_cell={
                "textAlign": "left",
                "padding": "12px",
                "fontSize": "13px",
                "color": "#111827",
                "borderBottom": "1px solid #f3f4f6",
                "whiteSpace": "normal",
                "height": "auto",
                "cursor": "pointer",
                "minWidth": "80px",
                "maxWidth": "200px",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#f9fafb"},
                {"if": {"column_id": "progress"}, "fontWeight": "800", "color": "#7c3aed"},
            ],
            page_size=10,
            sort_action="native",
            filter_action="native",
        )
        ])
    max_date = scope_df["date"].max() if "date" in scope_df.columns else pd.Timestamp(datetime.now().date())
    if date_range == "custom" and custom_start and custom_end:
        min_date = pd.Timestamp(pd.to_datetime(custom_start).date())
        max_date = pd.Timestamp(pd.to_datetime(custom_end).date())
    else:
        days = int(date_range) if date_range != "custom" else 30
        min_date = max_date - timedelta(days=days - 1)

    fig, days_shown, total_days = create_session_timeline(scope_df, min_date, max_date, is_all_users)
    date_range_text = f"Showing {days_shown} of {total_days} days"

    words_uploaded_display = f"{content_metrics['total_words_uploaded']:,}" if isinstance(content_metrics["total_words_uploaded"], int) else content_metrics["total_words_uploaded"]

    return (
        first_event_str,
        str(total_sessions),
        f"{active_days} days",
        last_seen_str,
        target_lang,
        native_lang,
        user_city,
        user_device,
        top_event,
        f"{total_events:,}",
        str(content_metrics["total_contents"]),
        words_uploaded_display,
        content_metrics["total_words_completed"],
        f"{content_metrics['avg_progress']}%",
        f"{content_metrics['avg_playback_time']} min",
        content_table,
        fig,
        date_range_text,
        user_id,
    )

@app.callback(
    [
        Output("selected-content-store", "data"),
        Output("content-table", "style_data_conditional"),
    ],
    [
        Input("content-table", "active_cell"),
        Input("content-table", "data"),
    ],
    prevent_initial_call=False,
)
def handle_content_row_click(active_cell, table_data):
    base_styles = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#f9fafb"},
        {"if": {"column_id": "progress"}, "fontWeight": "800", "color": "#7c3aed"},
    ]

    if not table_data or active_cell is None:
        return None, base_styles

    row = active_cell.get("row")
    if row is None or row >= len(table_data):
        return None, base_styles

    selected_content_id = table_data[row].get("content_id")
    if not selected_content_id:
        return None, base_styles

    highlight_style = base_styles + [{
        "if": {"filter_query": f'{{content_id}} = "{selected_content_id}"'},
        "backgroundColor": "#ede9fe",
    }]

    return selected_content_id, highlight_style

@app.callback(
    Output("content-details-container", "children"),
    [
        Input("selected-content-store", "data"),
        Input("current-user-store", "data"),
        Input("content-date-filter-store", "data"),
        Input("env-store", "data"),
    ],
    prevent_initial_call=False,
)
def show_content_details(selected_content_id, current_user_value, date_filter, env_value):
    if not selected_content_id:
        return html.Div()

    if current_user_value == ALL_USERS_VALUE:
        scope_df = df.copy()
    else:
        distinct_id_col = _pick_col(df, ["properties.distinct_id", "distinct_id"])
        if distinct_id_col:
            scope_df = df[df[distinct_id_col].astype(str) == str(current_user_value)].copy()
        else:
            scope_df = df.copy()

    history_df, summary = get_content_history(scope_df, selected_content_id, date_filter)
    if len(history_df) == 0:
        return html.Div()

    details = ella.fetch_content_details(env_value, selected_content_id)
    content_title = details.get("title", "Content")

    fig_progress = go.Figure()
    fig_progress.add_trace(go.Scatter(
        x=history_df["date"],
        y=history_df["max_progress"],
        mode="lines+markers",
        line=dict(color="#7c3aed", width=2),
        marker=dict(size=8, color="#7c3aed"),
        hovertemplate="<b>%{x|%b %d}</b><br>Progress: %{y:.1f}%<extra></extra>",
    ))
    fig_progress.update_layout(
        title="Progress over time",
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickformat="%b %d", title="Date"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", rangemode="tozero", title="Progress (%)"),
        margin=dict(l=60, r=20, t=40, b=50),
        hovermode="x unified",
    )

    fig_playback = go.Figure()
    fig_playback.add_trace(go.Scatter(
        x=history_df["date"],
        y=history_df["playback_time_minutes"],
        mode="lines+markers",
        line=dict(color="#22c55e", width=2),
        marker=dict(size=8, color="#22c55e"),
        hovertemplate="<b>%{x|%b %d}</b><br>Playback: %{y:.1f} min<extra></extra>",
    ))
    fig_playback.update_layout(
        title="Playback time over time",
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickformat="%b %d", title="Date"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", rangemode="tozero", title="Playback time (min)"),
        margin=dict(l=60, r=20, t=40, b=50),
        hovermode="x unified",
    )

    date_buttons = html.Div([
        html.Span("Show: ", style={"marginRight": "10px", "fontWeight": "800", "fontSize": "12px", "color": "#374151"}),
        dbc.ButtonGroup([
            dbc.Button("Last week", id={"type": "content-filter-btn", "value": "week"}, size="sm", outline=True, color="primary", active=(date_filter == "week")),
            dbc.Button("Last month", id={"type": "content-filter-btn", "value": "month"}, size="sm", outline=True, color="primary", active=(date_filter == "month")),
        ])
    ], style={"marginBottom": "14px"})

    summary_cards = dbc.Row([
        dbc.Col(html.Div([
            html.Div("Overall max progress", style={"fontSize": "12px", "color": "#7c3aed", "fontWeight": "800"}),
            html.Div(f"{summary['overall_max_progress']}%", style={"fontSize": "26px", "fontWeight": "900", "color": "#111827"}),
        ], style={"backgroundColor": "#f3e8ff", "padding": "16px", "borderRadius": "12px"}), width=3),
        dbc.Col(html.Div([
            html.Div("Total playback", style={"fontSize": "12px", "color": "#16a34a", "fontWeight": "800"}),
            html.Div(f"{summary['total_playback_time_minutes']} min", style={"fontSize": "26px", "fontWeight": "900", "color": "#111827"}),
        ], style={"backgroundColor": "#d1fae5", "padding": "16px", "borderRadius": "12px"}), width=3),
        dbc.Col(html.Div([
            html.Div("Days interacted", style={"fontSize": "12px", "color": "#f59e0b", "fontWeight": "800"}),
            html.Div(f"{summary['days_interacted']}", style={"fontSize": "26px", "fontWeight": "900", "color": "#111827"}),
        ], style={"backgroundColor": "#fef3c7", "padding": "16px", "borderRadius": "12px"}), width=3),
        dbc.Col(html.Div([
            html.Div("Period", style={"fontSize": "12px", "color": "#64748b", "fontWeight": "800"}),
            html.Div(f"{summary['total_days']} days", style={"fontSize": "26px", "fontWeight": "900", "color": "#111827"}),
        ], style={"backgroundColor": "#f1f5f9", "padding": "16px", "borderRadius": "12px"}), width=3),
    ], className="g-2", style={"marginBottom": "14px"})

    graphs_row = dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_progress, config={"displayModeBar": False}), width=6),
        dbc.Col(dcc.Graph(figure=fig_playback, config={"displayModeBar": False}), width=6),
    ], className="g-2")

    words = list_saved_words_for_content(scope_df, selected_content_id)
    saved_words_section = html.Div()
    if words:
        chips = [html.Span(
            w,
            style={
                "display": "inline-block",
                "padding": "6px 10px",
                "margin": "6px 6px 0 0",
                "borderRadius": "999px",
                "backgroundColor": "#f3f4f6",
                "color": "#111827",
                "fontSize": "12px",
                "fontWeight": "700",
            }
        ) for w in words]

        saved_words_section = html.Div([
            html.Hr(style={"margin": "16px 0", "borderColor": "#e5e7eb"}),
            html.Div(f"Saved words ({len(words)})", style={"fontWeight": "900", "fontSize": "12px", "color": "#111827", "marginBottom": "8px"}),
            html.Div(chips),
        ])

    return html.Div([
        html.Div([
            html.Div(f"Content details: {content_title[:80]}", style={"fontSize": "16px", "fontWeight": "900", "color": "#111827", "marginBottom": "14px"}),
            summary_cards,
            date_buttons,
            graphs_row,
            saved_words_section,
        ], style=card_style())
    ])

@app.callback(
    Output("content-date-filter-store", "data"),
    [Input({"type": "content-filter-btn", "value": "week"}, "n_clicks"),
     Input({"type": "content-filter-btn", "value": "month"}, "n_clicks")],
    prevent_initial_call=True,
)
def update_content_date_filter(week_clicks, month_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return "month"
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if "week" in button_id:
        return "week"
    return "month"

@app.callback(
    Output("session-details-container", "children"),
    [Input("session-timeline", "clickData")],
    [State("current-user-store", "data"), State("env-store", "data")],
    prevent_initial_call=True,
)
def display_session_details(clickData, current_user_value, env_value):
    if clickData is None:
        return dash.no_update

    clicked_date = pd.Timestamp(pd.to_datetime(clickData["points"][0]["x"]).date())

    if current_user_value == ALL_USERS_VALUE:
        scope_df = df.copy()
    else:
        distinct_id_col = _pick_col(df, ["properties.distinct_id", "distinct_id"])
        if distinct_id_col:
            scope_df = df[df[distinct_id_col].astype(str) == str(current_user_value)].copy()
        else:
            scope_df = df.copy()

    sessions = get_session_details_with_content(env_value, scope_df, clicked_date)

    if len(sessions) == 0:
        return html.Div([
            html.Div([
                html.Div(f"{clicked_date.strftime('%b %d, %Y')} sessions", style={"fontSize": "16px", "fontWeight": "900", "color": "#111827", "marginBottom": "8px"}),
                html.Div("No sessions on this date.", style={"color": "#6b7280", "fontSize": "13px"}),
            ], style=card_style())
        ])

    total_sessions = len(sessions)
    total_time = sum([s["duration"] for s in sessions])
    avg_duration = total_time / total_sessions if total_sessions > 0 else 0

    accordion_items = []
    for idx, session in enumerate(sessions, 1):
        top_events_html = []
        for event_name, count in session["top_events"].items():
            top_events_html.append(html.Div([
                html.Span(f"{event_name}: ", style={"color": "#6b7280", "fontSize": "12px", "fontWeight": "700"}),
                html.Span(f"{count}", style={"fontWeight": "900", "fontSize": "12px", "color": "#7c3aed"}),
            ], style={"marginBottom": "4px"}))

        content_html = []
        if session["content_played"]:
            for content in session["content_played"]:
                listen_url = f"https://helloella-prod.web.app/listen/{content['content_id']}"
                change = content["progress_change"]
                if change > 0:
                    change_color = "#16a34a"
                    change_symbol = "+"
                elif change < 0:
                    change_color = "#ef4444"
                    change_symbol = ""
                else:
                    change_color = "#64748b"
                    change_symbol = ""

                content_html.append(html.Div([
                    html.A(
                        content["title"][:60] + ("..." if len(content["title"]) > 60 else ""),
                        href=listen_url,
                        target="_blank",
                        style={"color": "#7c3aed", "textDecoration": "none", "fontWeight": "900", "fontSize": "13px"}
                    ),
                    html.Div([
                        html.Span(f"Progress: {content['old_progress']}% → {content['new_progress']}% ", style={"color": "#6b7280", "fontSize": "12px", "fontWeight": "700"}),
                        html.Span(f"({change_symbol}{change}%)", style={"color": change_color, "fontSize": "12px", "fontWeight": "900"}),
                        html.Span(f" | Playback: {content['playback_time_min']} min", style={"color": "#6b7280", "fontSize": "12px", "fontWeight": "700"}),
                    ]),
                ], style={"marginBottom": "10px"}))

        accordion_items.append(dbc.AccordionItem(
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([html.Span("Time: ", style={"color": "#6b7280", "fontSize": "12px", "fontWeight": "700"}),
                                  html.Span(f"{session['start_time']}–{session['end_time']}", style={"fontWeight": "900", "fontSize": "12px"})], style={"marginBottom": "6px"}),
                        html.Div([html.Span("Duration: ", style={"color": "#6b7280", "fontSize": "12px", "fontWeight": "700"}),
                                  html.Span(f"{session['duration']} min", style={"fontWeight": "900", "fontSize": "12px"})]),
                    ], width=6),
                    dbc.Col([
                        html.Div([html.Span("Total events: ", style={"color": "#6b7280", "fontSize": "12px", "fontWeight": "700"}),
                                  html.Span(f"{session['event_count']}", style={"fontWeight": "900", "fontSize": "12px"})], style={"marginBottom": "6px"}),
                        html.Div([html.Span("Event types: ", style={"color": "#6b7280", "fontSize": "12px", "fontWeight": "700"}),
                                  html.Span(f"{session['event_types']}", style={"fontWeight": "900", "fontSize": "12px"})]),
                    ], width=6),
                ], className="g-2"),
                html.Hr(style={"margin": "10px 0", "borderColor": "#e5e7eb"}),

                html.Div("Top events", style={"fontWeight": "900", "fontSize": "12px", "color": "#111827", "marginBottom": "8px"}),
                html.Div(top_events_html),

                html.Hr(style={"margin": "10px 0", "borderColor": "#e5e7eb"}),

                html.Div("Content played", style={"fontWeight": "900", "fontSize": "12px", "color": "#111827", "marginBottom": "8px"}),
                html.Div(content_html if content_html else html.Div("No content played.", style={"color": "#6b7280", "fontSize": "12px"})),
            ]),
            title=f"Session {idx} | {session['start_time']}–{session['end_time']}, {session['duration']} min | {len(session['content_played'])} content(s)"
        ))

    return html.Div([
        html.Div([
            html.Div(f"{clicked_date.strftime('%b %d, %Y')} sessions", style={"fontSize": "16px", "fontWeight": "900", "color": "#111827", "marginBottom": "10px"}),
            html.Div([
                html.Span(f"Sessions: {total_sessions}", style={"marginRight": "16px", "fontSize": "12px", "color": "#6b7280", "fontWeight": "800"}),
                html.Span(f"Total time: {total_time} min", style={"marginRight": "16px", "fontSize": "12px", "color": "#6b7280", "fontWeight": "800"}),
                html.Span(f"Avg duration: {int(avg_duration)} min", style={"fontSize": "12px", "color": "#6b7280", "fontWeight": "800"}),
            ], style={"marginBottom": "10px"}),
            dbc.Accordion(accordion_items, start_collapsed=True, flush=True),
        ], style=card_style()),
    ])
