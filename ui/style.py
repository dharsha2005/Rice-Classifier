from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink:#19323a; --muted:#61777a; --teal:#087f7b; --teal-dark:#075b5d; --mint:#e8f4f0; --coral:#e77c5b; --cream:#f7f6f0; --line:#dce7e3; }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
        .stApp { background:var(--cream); background-image:linear-gradient(135deg,rgba(232,244,240,.75),rgba(247,246,240,.96) 42%,rgba(255,255,255,.9)); }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stSidebar"] { background:#123f45; border-right:1px solid rgba(255,255,255,.08); }
        [data-testid="stSidebar"] * { color:#edf8f4 !important; }
        [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.18); }
        h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink) !important; letter-spacing:0 !important; }
        h1 { font-size:2.4rem !important; }
        .hero { padding:2rem 2.25rem; margin:.6rem 0 1.7rem; border-radius:18px; color:white; background:linear-gradient(115deg,#075b5d,#087f7b 58%,#2b9b8f); box-shadow:0 14px 34px rgba(7,91,93,.18); }
        .hero-kicker { color:#b9e7dc; font:700 .74rem 'Space Grotesk',sans-serif; letter-spacing:.15em; text-transform:uppercase; margin-bottom:.65rem; }
        .hero h1 { color:white !important; font-size:clamp(2rem,4vw,3.2rem) !important; line-height:1.05; margin:0 0 .7rem; }
        .hero p { color:#e3f5f0; margin:0; max-width:720px; }
        .eyebrow { color:var(--teal); font:700 .72rem 'Space Grotesk',sans-serif; letter-spacing:.14em; text-transform:uppercase; }
        .section-note { color:var(--muted); margin-top:-.45rem; }
        .stButton > button, .stDownloadButton > button { border-radius:9px; border:1px solid var(--teal); background:white; color:var(--teal-dark); font-weight:700; }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--coral); color:#9c4931; box-shadow:0 5px 14px rgba(7,91,93,.12); }
        .stButton > button[kind="primary"] { background:var(--coral); border-color:var(--coral); color:white; }
        [data-testid="stMetric"] { background:rgba(255,255,255,.78); border:1px solid var(--line); border-radius:12px; padding:.75rem 1rem; }
        [data-testid="stFileUploader"] { background:rgba(255,255,255,.72); border:1px dashed #9cc8bf; border-radius:12px; padding:.35rem; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
        [data-testid="stAlert"] { border-radius:10px; }
        hr { border-color:var(--line); }
        .status-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; background:#36a269; }
        .status-dot.warn { background:#e7a84b; }
        .surface { background:rgba(255,255,255,.72); border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, description: str, kicker: str = "Rice quality intelligence") -> None:
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">{kicker}</div><h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def page_heading(title: str, description: str) -> None:
    st.markdown(f'<div class="eyebrow">Rice quality intelligence</div><h1>{title}</h1><p class="section-note">{description}</p>', unsafe_allow_html=True)
