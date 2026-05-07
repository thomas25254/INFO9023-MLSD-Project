import os

import pandas as pd
import plotly.graph_objects as go
import requests
from google.cloud import bigquery

import streamlit as st

# Config
API_URL = os.environ.get(
    "API_URL", "https://hearedit-api-726024632692.europe-west1.run.app"
)
BQ_TABLE = os.environ.get(
    "BQ_TABLE", "info9023-project-hearedit.hearedit_dataset.transcriptions"
)

st.set_page_config(
    page_title="HearEdit — Audio Transcription",
    page_icon="🎙️",
    layout="centered",
)

# CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    background: #f4f5f7;
    color: #1a1a2e;
}
.stApp { background: #f4f5f7; }

.app-header { text-align: center; margin-bottom: 2rem; }
.app-header h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; color: #1a1a2e; }
.app-header p { color: #555; font-size: 1rem; margin-top: 0.4rem; }

.card {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 1rem;
}

.fulltext-box {
    background: #f9f9ff;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    line-height: 1.75;
    font-size: 0.95rem;
    color: #1a1a2e;
    border: 1px solid #ebebf5;
}

.spk-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
}

.seg-row {
    display: flex;
    gap: 0.8rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid #f0f0f0;
    align-items: flex-start;
}
.seg-row:last-child { border-bottom: none; }

.seg-time {
    font-family: monospace;
    font-size: 0.8rem;
    color: #888;
    white-space: nowrap;
    padding-top: 0.15rem;
    min-width: 95px;
}

.seg-text { font-size: 0.95rem; line-height: 1.6; color: #1a1a2e; flex: 1; }

.stButton > button {
    background: #4f46e5 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.55rem 2rem !important;
    width: 100% !important;
}
.stButton > button:hover { background: #4338ca !important; }
.stButton > button:disabled { background: #a5b4fc !important; }

[data-testid="stMetric"] {
    background: #fff;
    border-radius: 10px;
    padding: 1rem !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    text-align: center;
}
[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; color: #4f46e5 !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #888 !important; text-transform: uppercase; letter-spacing: 0.05em; }

[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #ebebf5; }

/* Expander styling */
[data-testid="stExpander"] {
    background: #fff;
    border: 1px solid #ebebf5 !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### HearEdit")
    st.markdown(
        "<div style='color:#888; font-size:0.85rem; margin-bottom:1.5rem;'>Audio Transcription</div>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation", ["Transcribe", "Historical"], label_visibility="collapsed"
    )

# Helpers
COLORS = ["#4f46e5", "#0891b2", "#059669", "#dc2626", "#d97706", "#7c3aed", "#db2777"]


def get_color(speakers_list, spk):
    idx = list(speakers_list).index(spk) if spk in speakers_list else 0
    return COLORS[idx % len(COLORS)]


def fmt_time(s):
    return f"{int(s // 60)}:{s % 60:05.2f}"


# PAGE 1 — Transcribe
if page == "Transcribe":
    st.markdown(
        """
    <div class="app-header">
        <h1>HearEdit</h1>
        <p>Upload an audio file and get its transcription with speaker identification.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    audio_file = st.file_uploader(
        "Audio", type=["wav", "mp3", "ogg", "flac"], label_visibility="collapsed"
    )
    run = st.button("Transcribe", disabled=not audio_file)

    if audio_file and run:
        with st.spinner("Transcribing, please wait…"):
            try:
                audio_file.seek(0)
                resp = requests.post(
                    f"{API_URL}/transcribe",
                    files={"audio": (audio_file.name, audio_file, "audio/wav")},
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"API error: {e}")
                st.stop()

        segments = data.get("segments", [])
        if not segments:
            st.warning("No segments returned.")
            st.stop()

        df = pd.DataFrame(segments)
        df["duration"] = df["end"] - df["start"]
        speakers = df["speaker"].unique().tolist()

        # Métriques
        c1, c2, c3 = st.columns(3)
        c1.metric("Segments", len(df))
        c2.metric("Speakers", len(speakers))
        c3.metric("Words", len(data.get("full_text", "").split()))

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Full text
        st.markdown(
            '<div class="section-title">Full text</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="fulltext-box">{data.get("full_text", "")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Pie chart
        st.markdown(
            '<div class="section-title">Speaker breakdown</div>', unsafe_allow_html=True
        )
        spk_time = df.groupby("speaker")["duration"].sum().reset_index()
        colors_list = [get_color(speakers, s) for s in spk_time["speaker"]]

        fig_pie = go.Figure(
            go.Pie(
                labels=spk_time["speaker"],
                values=spk_time["duration"].round(1),
                hole=0.5,
                marker={"colors": colors_list, "line": {"color": "#fff", "width": 2}},
                textfont={"family": "Inter", "size": 13},
                hovertemplate="<b>%{label}</b><br>%{value:.1f}s (%{percent})<extra></extra>",
            )
        )
        fig_pie.add_annotation(
            text=f"<b>{len(speakers)}</b><br>speaker(s)",
            x=0.5,
            y=0.5,
            font={"family": "Inter", "size": 13, "color": "#555"},
            showarrow=False,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend={"font": {"family": "Inter"}, "bgcolor": "rgba(0,0,0,0)"},
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            height=240,
        )
        st.plotly_chart(fig_pie, width=700)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Timeline Gantt
        st.markdown('<div class="section-title">Timeline</div>', unsafe_allow_html=True)
        fig_gantt = go.Figure()
        for spk in speakers:
            color = get_color(speakers, spk)
            spk_df = df[df["speaker"] == spk]
            for _, row in spk_df.iterrows():
                fig_gantt.add_trace(
                    go.Bar(
                        x=[row["duration"]],
                        y=[row["speaker"]],
                        base=[row["start"]],
                        orientation="h",
                        marker={"color": color, "opacity": 0.85, "line": {"width": 0}},
                        hovertemplate=(
                            f"<b>{spk}</b><br>"
                            f"{fmt_time(row['start'])} → {fmt_time(row['end'])}<br>"
                            f"<i>{row['text'][:60]}{'…' if len(row['text']) > 60 else ''}</i>"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

        fig_gantt.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f9f9ff",
            font={"family": "Inter", "color": "#555", "size": 12},
            xaxis={
                "title": "Time (s)",
                "gridcolor": "#ebebf5",
                "zerolinecolor": "#ebebf5",
            },
            yaxis={"gridcolor": "#ebebf5"},
            margin={"l": 0, "r": 0, "t": 10, "b": 40},
            height=max(100, len(speakers) * 60 + 60),
        )
        st.plotly_chart(fig_gantt, width=700)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Segments
        st.markdown('<div class="section-title">Segments</div>', unsafe_allow_html=True)

        for _, row in df.iterrows():
            color = get_color(speakers, row["speaker"])
            st.markdown(
                f"""
            <div class="seg-row">
                <div>
                    <span class="spk-badge" style="background:{color}18; color:{color}; border:1px solid {color}40;">
                        {row["speaker"]}
                    </span>
                    <div class="seg-time">{fmt_time(row["start"])} → {fmt_time(row["end"])}</div>
                </div>
                <div class="seg-text">{row["text"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# PAGE 2 —  historical
elif page == "Historical":
    st.markdown(
        """
    <div class="app-header">
        <h1> Historical</h1>
        <p>Past transcriptions stored in BigQuery.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    @st.cache_data(ttl=60)
    def load_history():
        client = bigquery.Client()
        query = f"""
            SELECT filename, full_text, num_segments, created_at
            FROM `{BQ_TABLE}`
            ORDER BY created_at DESC
            LIMIT 100
        """
        return client.query(query).to_dataframe()

    try:
        df_history = load_history()
    except Exception as e:
        st.error(f"Cannot load from BigQuery: {e}")
        st.stop()

    if df_history.empty:
        st.info("No transcriptions recorded yet.")
    else:
        # Métriques
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(df_history))
        c2.metric("Total segments", int(df_history["num_segments"].sum()))
        c3.metric("Avg segments", f"{df_history['num_segments'].mean():.1f}")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Transcriptions par jour
        st.markdown(
            '<div class="section-title">Transcriptions per day</div>',
            unsafe_allow_html=True,
        )
        df_history["date"] = pd.to_datetime(df_history["created_at"]).dt.date
        daily = df_history.groupby("date").size().reset_index(name="count")

        fig_bar = go.Figure(
            go.Bar(
                x=daily["date"].astype(str),
                y=daily["count"],
                marker={"color": "#4f46e5", "opacity": 0.8, "line": {"width": 0}},
                hovertemplate="<b>%{x}</b><br>%{y} transcription(s)<extra></extra>",
            )
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#f9f9ff",
            font={"family": "Inter", "color": "#555"},
            xaxis={"gridcolor": "#ebebf5"},
            yaxis={"gridcolor": "#ebebf5", "title": "Count"},
            margin={"l": 0, "r": 0, "t": 10, "b": 40},
            height=220,
        )
        st.plotly_chart(fig_bar, width=700)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Liste avec expanders — un par transcription
        st.markdown(
            '<div class="section-title">All transcriptions</div>',
            unsafe_allow_html=True,
        )

        for _, row in df_history.iterrows():
            created = pd.to_datetime(row["created_at"]).strftime("%Y-%m-%d %H:%M")
            label = f"📄 **{row['filename']}** — {created} — {row['num_segments']} segment(s)"
            with st.expander(label):
                st.markdown(
                    f'<div class="fulltext-box">{row["full_text"]}</div>',
                    unsafe_allow_html=True,
                )
