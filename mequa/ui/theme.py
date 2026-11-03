"""Mequa workspace theme: "Minimal Striking" (light, hairlines, one mint accent).

White canvas, hairline #EDEDED rules, light-weight large numerals, tiny tracked mono labels, black
primary buttons and a single mint (#0AFFD2) accent for state: the look of the supplied
"Minimal Striking Professional" reference. Fonts are system fonts so it looks the same offline.

apply() injects the CSS (cheap, safe on every rerun). page_header() opens each page with a compact
title strip. The other helpers return small HTML pieces.
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence, Tuple

import streamlit as st

VERSION = "6.3.0"

INK = "#0A0A0A"         # text, primary buttons
CANVAS = "#FFFFFF"      # app background
PANEL = "#FCFCFC"       # raised surface
PANEL_2 = "#F5F5F5"     # chips, hover
LINE = "#EDEDED"        # hairlines
MUTED = "#8A8A8A"
MINT = "#0AFFD2"        # the one accent
ACCENT = MINT
ACCENT_DEEP = "#00C9A7"
TEXT = INK
OK = "#16A34A"
WARN = "#CA8A04"
BAD = "#EF4444"

# kept so older imports keep working
NAVY, NAVY_2, INK_2 = PANEL, PANEL_2, CANVAS
RED, RED_HOT, RED_DEEP = BAD, BAD, BAD

_NAMES = {"julia": "Julia", "rust": "Rust", "numpy": "NumPy", "sam": "SAM", "classical": "Classical",
          "classical_split": "Classical (split)", "zone": "Zone", "strain": "Strain",
          "failure probability": "Failure Probability", "stiffness": "Stiffness", "group": "Group",
          "selection": "Selection"}


def nice_name(s: str) -> str:
    """Capitalised display name for option values like 'julia' or 'failure probability'."""
    return _NAMES.get(s, s[:1].upper() + s[1:] if s else s)


PALETTES = {
    "light": dict(bg="#FFFFFF", bg_a="rgba(255,255,255,.92)", panel="#FAFAF9", soft="#F4F4F2", line="#E7E7E4", ink="#111827", mute="#6B7280", faint="#9CA3AF",
                  dot="#C4C4C4", dot2="#D6D6D6", text2="#404040", mint="#0F766E", mint_deep="#0B5F58", on_accent="#FFFFFF", btn_bg="#111827", btn_fg="#FFFFFF",
                  shadow="rgba(0,0,0,.04)", ring="rgba(15,118,110,.20)"),
    "green": dict(bg="#060A08", bg_a="rgba(6,10,8,.92)", panel="#0A100D", soft="#111A15", line="#1E2B23", ink="#E6F2EA", mute="#7C9185", faint="#4F6358",
                  dot="#34463B", dot2="#2A3A31", text2="#BCD0C4", mint="#22E58B", mint_deep="#14B86D", on_accent="#02140B", btn_bg="#22E58B", btn_fg="#02140B",
                  shadow="rgba(0,0,0,.45)", ring="rgba(34,229,139,.28)"),
}
THEME_LABELS = {"light": "Light", "green": "Green / Black"}


def _vars(P: dict) -> str:
    return " ".join(f"--{k.replace('_', '-')}:{v};" for k, v in P.items()) + f" --ok:{OK}; --warn:{WARN}; --bad:{BAD};"


def current_mode() -> str:
    """'light' or 'green'. Session choice first, then the saved setting."""
    try:
        m = st.session_state.get("theme_mode")
        if m in PALETTES:
            return m
        from mequa import settings
        m = settings.load().get("theme", "light")
        return m if m in PALETTES else "light"
    except Exception:
        return "light"


def _dark_overrides() -> str:
    """Streamlit's own widgets are themed by config.toml (light). In the green theme they are repainted here."""
    return """
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stMarkdown, [data-testid="stMarkdownContainer"] { color:var(--ink); }
header[data-testid="stHeader"] { background:var(--bg-a); }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"] button { color:var(--mute); }
[data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="popover"] ul, [role="listbox"] { background:var(--panel) !important; color:var(--ink) !important; border:1px solid var(--line); }
[role="option"], [data-baseweb="menu"] li { background:transparent !important; color:var(--ink) !important; }
[role="option"][aria-selected="true"], [role="option"]:hover, [data-baseweb="menu"] li:hover { background:var(--soft) !important; }
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"], [data-testid="stNumberInput"] [role="group"], [data-testid="stTextInput"] [role="group"], [data-testid="stTextArea"] [role="group"], [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { background:var(--panel) !important; border-color:var(--line) !important; }
[data-testid="stSelectbox"] input, [data-testid="stSelectbox"] button, [data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] input, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { color:var(--ink) !important; -webkit-text-fill-color:var(--ink); fill:var(--ink); }
[data-testid="stTextArea"] textarea { background:var(--panel) !important; }
[data-baseweb="select"] svg, [data-baseweb="select"] span, [data-baseweb="input"] input, .stTextArea textarea, .stNumberInput input { color:var(--ink) !important; -webkit-text-fill-color:var(--ink); }
[data-testid="stChatInput"] textarea, [data-testid="stChatInput"] { background:var(--bg) !important; color:var(--ink) !important; }
[data-testid="stAlert"], [data-testid="stAlertContainer"] { background:var(--panel) !important; color:var(--ink) !important; }
[data-testid="stAlert"] * { color:var(--ink) !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { filter:invert(1) hue-rotate(180deg); }
[data-testid="stPlotlyChart"] .main-svg { background:transparent !important; }
.js-plotly-plot .gridlayer path, .js-plotly-plot .zerolinelayer path { stroke:var(--line) !important; stroke-opacity:.6; }
[data-testid="stTooltipContent"], [data-baseweb="tooltip"] > div { background:var(--panel) !important; color:var(--ink) !important; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--mint) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color:var(--mute); }
[data-testid="stCheckbox"] span[data-baseweb="checkbox"] > div, [data-testid="stToggle"] > label > div { border-color:var(--mute); }
.st-key-chat_scroll [data-testid="stChatMessageAvatarAssistant"] { background:var(--mint); color:var(--on-accent); border-color:var(--mint); }
code, [data-testid="stMarkdownContainer"] code { background:var(--soft) !important; color:var(--mint) !important; }
[data-testid="stButtonGroup"] button { background:var(--bg) !important; color:var(--mute) !important; border-color:var(--line) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"] { background:var(--mint) !important; color:var(--on-accent) !important; border-color:var(--mint) !important; }
[data-testid="stPopoverButton"], button[data-testid="stPopoverButton"] { background:var(--bg) !important; color:var(--ink) !important; border:1px solid var(--line) !important; }
[data-testid="stPopoverButton"] svg, [data-testid="stPopoverButton"] span { color:var(--ink) !important; fill:var(--ink); }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] div[data-baseweb] { background:var(--bg) !important; border-color:var(--line) !important; }
[data-testid="stChatInput"] textarea::placeholder { color:var(--mute) !important; -webkit-text-fill-color:var(--mute); }
[data-testid="stChatInput"] textarea { background:transparent !important; }
[data-testid="stFileUploader"] section { background:var(--panel); }

.mq-card:hover { border-color:var(--mint); }
[data-testid="stSlider"], [data-testid="stSelectSlider"], [data-testid="stRadio"], [data-testid="stCheckbox"], [data-testid="stToggle"], [data-testid="stFileChip"] { filter:invert(1) hue-rotate(180deg); }
[data-testid="stSlider"] *, [data-testid="stSelectSlider"] *, [data-testid="stRadio"] *, [data-testid="stCheckbox"] *, [data-testid="stToggle"] *, [data-testid="stFileChip"] * { color:#0A0A0A !important; }
[data-testid="stFileUploader"] button { background:var(--bg) !important; color:var(--ink) !important; border:1px solid var(--line) !important; }
[data-testid="stStatusWidget"], [data-testid="stStatus"], details[data-testid="stExpander"] summary, [data-testid="stExpander"] summary { background:var(--panel) !important; color:var(--ink) !important; }
[data-testid="stStatus"] { border:1px solid var(--line); border-radius:12px; }
"""


def _css(chat_open: bool, mode: str = "light") -> str:
    P = PALETTES.get(mode, PALETTES["light"])
    dock = 380
    right_pad = f"{dock + 28}px" if chat_open else "28px"
    chip_right = f"{dock + 20}px" if chat_open else "20px"
    return f"""
<style>
:root {{
  {_vars(P)} --ring-c:{P["ring"]};
  --mono:ui-monospace,'SFMono-Regular','Cascadia Mono',Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --ring:0 0 0 3px var(--ring-c);
}}
html, body, .stApp, [class*="css"] {{ font-family:var(--sans); font-size:14px; -webkit-font-smoothing:antialiased; }}
.stApp {{ background:var(--bg); color:var(--ink); }}
::selection {{ background:var(--mint); color:var(--on-accent); }}

/* ---------- top bar ---------- */
header[data-testid="stHeader"] {{ height:56px; min-height:56px; background:var(--bg-a); border-bottom:1px solid var(--line); }}
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {{ visibility:hidden; height:0; }}
[data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] {{ visibility:visible; color:var(--mute); }}
[data-testid="stApp"]:has(.mq-chip.post) .pre {{ display:none !important; }}
.mq-chip {{ position:fixed; top:12px; right:{chip_right}; z-index:1000002; display:flex; gap:10px; align-items:center; padding:6px 14px; border:1px solid var(--line);
  border-radius:999px; background:var(--bg); font:500 12px/1 var(--sans); color:var(--ink); white-space:nowrap; }}
.mq-chip .sep {{ color:var(--dot); }} .mq-chip .led {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--dot); margin-right:6px; }}
.mq-chip .led.ok {{ background:var(--mint); }} .mq-chip .led.warn {{ background:#facc15; }} .mq-chip .led.bad {{ background:var(--bad); }}
div:has(> .st-key-theme_switch) {{ position:fixed !important; top:10px; left:332px; z-index:1000003; width:auto !important; height:auto; margin:0 !important; }}
.st-key-theme_switch {{ width:auto !important; }}
.st-key-theme_switch button {{ min-height:30px; padding:4px 12px; font:600 12px/1 var(--sans); }}
.mq-brand {{ position:fixed; top:0; left:0; width:232px; height:56px; box-sizing:border-box; padding:21px 0 0 24px; background:var(--bg); z-index:1000002; font:800 17px/1 var(--sans); letter-spacing:-.01em; color:var(--ink); }}
.mq-brand small {{ font:400 13px/1 var(--sans); color:var(--mute); margin-left:8px; letter-spacing:0; }}

/* ---------- layout ---------- */
.block-container {{ max-width:none !important; padding:76px {right_pad} 64px 28px !important; }}
[data-testid="stMain"] {{ background:var(--bg); }}

/* ---------- sidebar and navigation ---------- */
section[data-testid="stSidebar"] {{ background:var(--bg); border-right:1px solid var(--line); }}
section[data-testid="stSidebar"] > div {{ background:var(--bg); }}
[data-testid="stSidebarHeader"] {{ height:56px; min-height:56px; padding:0 .7rem; }}
[data-testid="stSidebarContent"] {{ padding-bottom:40px; }}
[data-testid="stSidebarNav"] {{ padding:6px 0 8px; }}
[data-testid="stNavSectionHeader"], [data-testid="stSidebarNav"] span[class*="Header"] {{
  font:600 11px/1 var(--sans) !important; letter-spacing:.02em; color:var(--faint) !important; padding:18px 22px 6px;
}}
[data-testid="stSidebarNavLink"] {{ margin:0 8px; padding:6px 14px; min-height:28px; border-radius:8px; gap:10px; color:var(--mute) !important; background:transparent;
  font:500 14px/1.3 var(--sans); }}
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {{ font-size:18px; color:inherit; }}
[data-testid="stSidebarNavLink"] span:not([data-testid="stIconMaterial"]) {{ font:inherit !important; color:inherit !important; }}
[data-testid="stSidebarNavLink"]:hover {{ background:var(--soft); color:var(--ink) !important; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ color:var(--ink) !important; font-weight:700; background:var(--soft); }}
[data-testid="stSidebarNavLink"][aria-current="page"] [data-testid="stIconMaterial"] {{ color:var(--mint); }}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
  font:600 12px/1.2 var(--sans) !important; color:var(--mute); border-top:1px solid var(--line);
  padding:16px 0 4px; margin:6px 0 8px;
}}

/* ---------- type ---------- */
h1, h2, h3, h4 {{ color:var(--ink); font-family:var(--sans); letter-spacing:-.02em; }}
h1 {{ font-size:1.75rem !important; font-weight:650 !important; letter-spacing:-.02em; }}
h2, h3 {{ font-size:1.05rem !important; font-weight:600 !important; }}
[data-testid="stCaptionContainer"], .stCaption {{ color:var(--mute); font-size:12px; }}
label, .stRadio label, .stCheckbox label {{ color:var(--text2); font-size:13px; }}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {{
  border-radius:8px; font:600 13px/1 var(--sans); min-height:36px; padding:8px 16px;
  background:var(--bg); color:var(--ink); border:1px solid var(--line); box-shadow:none; transition:background .15s, border-color .15s, color .15s;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:var(--ink); background:var(--bg); color:var(--ink); }}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"], .stDownloadButton > button[kind="primary"] {{ background:var(--btn-bg); color:var(--btn-fg); border-color:var(--btn-bg); }}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{ background:var(--mint); color:var(--on-accent); border-color:var(--mint); }}
.stButton > button:focus-visible {{ outline:none; box-shadow:var(--ring); }}
.stButton > button:disabled {{ opacity:.35; }}

/* ---------- inputs ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stTextArea textarea, .stNumberInput input, .stTextInput input {{
  background:var(--bg) !important; border:1px solid var(--line) !important; border-radius:10px !important; color:var(--ink) !important; box-shadow:none !important;
}}
[data-baseweb="select"] > div:hover {{ border-color:var(--ink) !important; }}
.stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus, [data-baseweb="select"] > div:focus-within {{ border-color:var(--mint) !important; box-shadow:var(--ring) !important; }}
.stSlider [role="slider"] {{ background:var(--ink) !important; border:2px solid var(--bg); box-shadow:0 0 0 1px var(--ink); }}
.stSlider [data-baseweb="slider"] > div > div {{ background:var(--line); }}
[data-testid="stFileUploader"] section {{ background:var(--panel); border:1px dashed var(--dot2); border-radius:12px; }}
[data-testid="stFileUploader"] section:hover {{ border-color:var(--ink); }}
[data-testid="stRadio"] [role="radiogroup"] label span {{ font-size:13px; }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap:22px; border-bottom:1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ background:transparent; padding:8px 0; height:auto; font:500 14px/1.2 var(--sans); color:var(--mute); }}
.stTabs [data-baseweb="tab"]:hover {{ color:var(--ink); }}
.stTabs [aria-selected="true"] {{ color:var(--ink) !important; font-weight:700; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--mint) !important; height:2px; }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}

/* ---------- metrics read as large light numerals ---------- */
[data-testid="stMetric"] {{ background:transparent; border:0; border-top:1px solid var(--line); border-radius:0; padding:10px 0 4px; }}
[data-testid="stMetricLabel"] p {{ font:500 12px/1.3 var(--sans); color:var(--mute); white-space:normal; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {{ font:600 1.6rem/1.15 var(--sans) !important; letter-spacing:-.02em; color:var(--ink); white-space:normal !important; overflow:visible !important; }}
[data-testid="stExpander"] {{ background:var(--bg); border:1px solid var(--line); border-radius:12px; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
[data-testid="stAlert"] {{ border-radius:12px; border:1px solid var(--line); }}
code, pre {{ font-family:var(--mono) !important; }}
code {{ color:var(--ink); background:var(--soft); border-radius:5px; }}
hr {{ border-color:var(--line); }}
::-webkit-scrollbar {{ width:9px; height:9px; }} ::-webkit-scrollbar-thumb {{ background:var(--line); border-radius:6px; }} ::-webkit-scrollbar-thumb:hover {{ background:var(--mute); }}

/* ---------- components ---------- */
.mq-hero {{ padding:2px 0 18px; margin-bottom:14px; border-bottom:1px solid var(--line); }}
.mq-eyebrow {{ font:600 12px/1 var(--sans); color:var(--mint); }}
.mq-title {{ font:650 1.9rem/1.15 var(--sans); letter-spacing:-.025em; color:var(--ink); margin:8px 0 6px; }}
.mq-sub {{ color:var(--mute); font-size:14px; max-width:820px; line-height:1.5; }}
.mq-big {{ display:flex; align-items:baseline; gap:14px; padding:2px 0 14px; }}
.mq-big .dot {{ display:none; }}
.mq-big .num {{ font:650 3rem/1 var(--sans); letter-spacing:-.03em; }}
.mq-big .cap {{ color:var(--mute); font-size:15px; }}
.mq-big .right {{ margin-left:auto; font:600 14px/1 var(--sans); }}
.mq-pill {{ display:inline-flex; align-items:center; gap:7px; padding:5px 12px; border-radius:999px; font:500 12px/1.2 var(--sans); border:1px solid var(--line); background:var(--bg); color:var(--ink); margin:0 8px 8px 0; }}
.mq-dot {{ width:7px; height:7px; border-radius:50%; background:var(--dot); }}
.mq-pill.ok .mq-dot {{ background:var(--mint); }} .mq-pill.bad .mq-dot {{ background:var(--bad); }} .mq-pill.warn .mq-dot {{ background:#facc15; }}
.mq-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:14px; margin:6px 0 22px; }}
.mq-card {{ padding:16px 18px; border:1px solid var(--line); background:var(--bg); border-radius:12px; transition:border-color .15s; }}
.mq-card:hover {{ border-color:var(--mute); }}
.mq-group {{ font:600 12px/1 var(--sans); color:var(--mute); margin:22px 0 4px; }}
.mq-kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin:4px 0 8px; }}
.mq-kpi {{ border:1px solid var(--line); border-radius:12px; padding:14px 18px; background:var(--panel); }}
.mq-kpi .k {{ font:500 12px/1 var(--sans); color:var(--mute); }} .mq-kpi .v {{ font:600 1.15rem/1.3 var(--sans); margin-top:8px; }} .mq-kpi .s {{ font:400 12px/1.4 var(--sans); color:var(--mute); margin-top:2px; }}
.mq-card h4 {{ padding:0 !important; margin:0 0 4px !important; font-size:.98rem; font-weight:600; letter-spacing:-.01em; }}
.mq-card p {{ margin:0 !important; color:var(--mute); font-size:13px; line-height:1.5; }}
.mq-card .ic {{ font-size:20px; margin-bottom:8px; color:var(--mint); }}
.mq-num {{ display:inline-block; font:600 10px/1 var(--mono); letter-spacing:.16em; color:var(--mute); margin-bottom:10px; }}
.mequa-ok {{ border:1px solid var(--line); border-left:3px solid var(--mint); border-radius:10px; background:var(--bg); padding:10px 14px; font-weight:600; }}
.mequa-halt {{ border:1px solid var(--line); border-left:3px solid var(--bad); border-radius:10px; background:var(--bg); padding:10px 14px; font-weight:700; }}
.mequa-diag {{ border:1px solid var(--line); border-left:3px solid #facc15; border-radius:10px; background:var(--bg); padding:10px 14px; font:12.5px/1.45 var(--mono); }}
.mequa-logo {{ background:var(--bg); border-radius:12px; padding:6px; display:inline-block; border:1px solid var(--line); }}
.mq-view-label {{ font:600 12px/1 var(--sans); color:var(--mute); margin:2px 0 6px; }}
.mq-view-label b {{ color:var(--ink); }}

/* ---------- bottom status rail ---------- */
.mq-status {{ position:fixed; left:0; right:0; bottom:0; height:26px; z-index:1000001; display:flex; align-items:center; gap:22px; padding:0 20px 0 316px;
  background:var(--bg); border-top:1px solid var(--line); font:400 12px/1 var(--sans); color:var(--mute); white-space:nowrap; overflow:hidden; }}
.mq-status > span {{ flex:0 0 auto; }} .mq-status > span:nth-last-child(2) {{ flex:0 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; }}
.mq-status b {{ color:var(--ink); font-weight:600; }} .mq-status .sp {{ flex:1 1 0; min-width:8px; }}
.mq-status .led {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; background:var(--dot); vertical-align:middle; }}
.mq-status .led.ok {{ background:var(--mint); }} .mq-status .led.warn {{ background:#facc15; }} .mq-status .led.bad {{ background:var(--bad); }}

/* ---------- right agent dock ---------- */
.st-key-chat_dock {{ position:fixed; top:56px; right:0; bottom:26px; width:{dock}px; z-index:998; background:var(--bg); border-left:1px solid var(--line);
  display:flex; flex-direction:column; gap:0 !important; padding:0 !important; overflow:hidden; }}
.st-key-chat_dock > div {{ width:100%; }}
.st-key-chat_dock > div:has(> .st-key-chat_head), .st-key-chat_dock > div:has(> .st-key-chat_input) {{ flex:0 0 auto; }}
.st-key-chat_dock > div:has(> .st-key-chat_scroll) {{ flex:1 1 0; min-height:0; overflow:hidden; }}
.st-key-chat_head {{ border-bottom:1px solid var(--line); padding:14px 20px 12px; flex:0 0 auto; }}
.st-key-chat_head [data-testid="stHorizontalBlock"] {{ align-items:center; }}
.mq-chat-title {{ font:650 15px/1 var(--sans); }}
.mq-chat-title i {{ display:none; }}
.mq-chat-sub {{ font:400 12px/1 var(--sans); color:var(--mute); margin-top:6px; }}
.st-key-chat_scroll {{ height:100%; overflow-y:auto; overflow-x:hidden; padding:0 !important; gap:0 !important; }}
.st-key-chat_msgs {{ gap:0 !important; }}
.st-key-chat_scroll [data-testid="stChatMessage"] {{ padding:16px 20px; margin:0; border-radius:0; background:var(--bg); gap:12px; border-bottom:1px solid var(--line); }}
.st-key-chat_scroll [data-testid="stChatMessage"] p {{ font-size:14px; line-height:1.6; }}
.st-key-chat_scroll [data-testid="stChatMessageAvatarUser"], .st-key-chat_scroll [data-testid="stChatMessageAvatarAssistant"] {{
  width:24px; height:24px; border-radius:50%; font:700 11px/1 var(--sans); background:var(--bg); color:var(--ink); border:1px solid var(--line); }}
.st-key-chat_scroll [data-testid="stChatMessageAvatarAssistant"] {{ background:var(--ink); color:var(--btn-fg); border-color:var(--btn-bg); }}
.st-key-chat_input {{ flex:0 0 auto; border-top:1px solid var(--line); padding:10px 16px 12px; background:var(--bg); }}
.st-key-chat_input [data-testid="stChatInput"] {{ background:var(--bg); border:1px solid var(--line); border-radius:12px; }}
.st-key-chat_input [data-testid="stChatInput"]:focus-within {{ border-color:var(--mint); box-shadow:var(--ring); }}
.st-key-chat_input textarea {{ font-size:14px; }}
.st-key-chat_input button[data-testid="stChatInputSubmitButton"] {{ background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:50%; }}
.st-key-chat_input button[data-testid="stChatInputSubmitButton"]:hover {{ background:var(--mint); border-color:var(--mint); }}
.st-key-chat_open {{ position:fixed; right:20px; bottom:42px; z-index:998; width:auto; }}
.mq-empty {{ color:var(--mute); font-size:13px; line-height:1.6; padding:18px 20px; }}
{_dark_overrides() if mode == "green" else ""}
</style>
"""


def _register_plotly(mode: str = "light") -> None:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception:
        return
    P = PALETTES.get(mode, PALETTES["light"])
    dark = mode == "green"
    grid, zero = ("#14201A", "#1E2B23") if dark else ("#F0F0F0", "#E4E4E4")
    axis = dict(gridcolor=grid, zerolinecolor=zero, linecolor=P["line"], tickfont=dict(size=11, color=P["mute"]), title_font=dict(size=11, color=P["mute"]))
    scene_ax = dict(backgroundcolor=P["bg"], gridcolor=grid, zerolinecolor=zero, color=P["mute"])
    pio.templates["mequa"] = go.layout.Template(
        layout=dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=P["bg"],
            font=dict(family="ui-sans-serif, system-ui, Segoe UI, sans-serif", color=P["ink"], size=12),
            colorway=[P["ink"], P["mint"], "#3B82F6", "#EF4444", "#FACC15", "#9B7FD1", "#4CAF7A", "#8A8A8A"],
            xaxis=axis, yaxis=axis, scene=dict(xaxis=scene_ax, yaxis=scene_ax, zaxis=scene_ax),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=P["ink"])), margin=dict(l=40, r=20, t=36, b=36),
        )
    )
    pio.templates.default = "mequa"


def rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def ink() -> str:
    return PALETTES[current_mode()]["ink"]


def accent() -> str:
    return PALETTES[current_mode()]["mint"]


def apply(chat_open: bool = True) -> None:
    """Inject the theme. Safe to call on every rerun."""
    mode = current_mode()
    st.markdown(_css(chat_open, mode), unsafe_allow_html=True)
    _register_plotly(mode)


def page_header(title: str, subtitle: str = "", eyebrow: str = "Mequa") -> None:
    st.markdown(
        f'<div class="mq-hero"><div class="mq-eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="mq-title">{html.escape(title)}</div>'
        f'<div class="mq-sub">{html.escape(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def big_number(value: str, caption: str = "", right: str = "") -> None:
    """The hero readout: a large light numeral with a mint dot, a muted caption and an optional right-hand fact."""
    st.markdown(f'<div class="mq-big"><span class="dot"></span><span class="num">{html.escape(value)}</span>'
                f'<span class="cap">{html.escape(caption)}</span><span class="right">{html.escape(right)}</span></div>', unsafe_allow_html=True)


def pill(label: str, state: str = "idle") -> str:
    """Return HTML for a status pill; state is 'ok', 'bad', 'warn' or 'idle'."""
    return f'<span class="mq-pill {state}"><span class="mq-dot"></span>{html.escape(label)}</span>'


def stat_row(items: Iterable[Tuple[str, str]]) -> None:
    items = list(items)
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def cards(items: Sequence[Tuple[str, str, str]]) -> None:
    """Equal-height responsive grid of (icon, title, text) cards. icon is a Material symbol name."""
    body = "".join(
        f'<div class="mq-card"><span class="material-symbols-rounded ic" style="font-family:\'Material Symbols Rounded\';font-variation-settings:\'FILL\' 0">{html.escape(n)}</span>'
        f'<h4>{html.escape(t)}</h4><p>{html.escape(d)}</p></div>' for n, t, d in items)
    st.markdown(f'<div class="mq-grid">{body}</div>', unsafe_allow_html=True)


def kpis(items: Sequence[Tuple[str, str, str]]) -> None:
    """Row of (label, value, sub) summary tiles."""
    body = "".join(f'<div class="mq-kpi"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div><div class="s">{html.escape(sub)}</div></div>'
                   for k, v, sub in items)
    st.markdown(f'<div class="mq-kpis">{body}</div>', unsafe_allow_html=True)
