"""
Amazon Sponsored Products — Search Term Intelligence & Cannibalization Tool
-----------------------------------------------------------------------------
Analyzes Amazon Ads Search Term reports to surface:
  1. A performance dashboard (blue Spend-vs-ACOS and Sales-vs-ACOS charts on
     separate scales, Waste-Spend-by-match-type and Sales-by-match-type pies).
  2. Keyword cannibalization — the same search term triggering multiple
     campaign x ad-group combos, with a search-term filter and an overview of
     cannibalized-term count, waste clicks, waste spend, total sales and ACOS.
  3. Insights & recommendations — spend-with-no-sales candidates, a harvest ACOS
     cutoff, a match-type filter, and a merged efficiency pie.
  4. Portfolio Explorer — prefix-aware campaign selection; portfolio and campaign
     summaries carry Spend / Waste Spend / Sales / ACOS.
  5. Campaigns x Match Type — AUTO is a first-class match-type group; waste
     columns everywhere; leaf ACOS shown as a colored badge.
  6. Waste Clicks — portfolio-friendly breakdown of clicks that never converted,
     bucketed (1, 2, 3, 4, 5, 5-10, 10-50, 50+) with pies, then a drill-down.

"Waste" is defined at the report-row level: any Spend / Clicks on a row that
produced zero Sales. Summed up a level it answers "how much did I pay for
clicks that never converted".

Scope: only portfolios whose name contains "FBA" are considered, EXCLUDING
any portfolio that also contains "Vizari".

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import re
import html
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------------------------------------
# Visual theme — Google Material palette (Looker Studio look).
# ----------------------------------------------------------------------

PALETTE = {
    "blue": "#4285F4", "red": "#EA4335", "yellow": "#FBBC04",
    "green": "#34A853", "purple": "#A142F4", "teal": "#24C1E0",
}
COLORWAY = [PALETTE["blue"], PALETTE["green"], PALETTE["red"],
            PALETTE["yellow"], PALETTE["purple"], PALETTE["teal"]]

BLUE_SCALE = [[0.0, "#C6DAFC"], [0.5, "#6BA1F5"], [1.0, "#1A56C4"]]
RED_SCALE = [[0.0, "#FBD8D8"], [0.5, "#F07B72"], [1.0, "#C5221F"]]
ACOS_LINE = "#EA8600"
SPEND_BLUE = "#4285F4"
SALES_TEAL = "#24C1E0"
WASTE_RED = "#EA4335"

CSS = """
<style>
html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
.stApp { background-color: #F8F9FA; }
h1, h2, h3 { font-family: 'Roboto', sans-serif; font-weight: 500; color: #202124; }

.kpi-card {
    background: #FFFFFF; border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(60,64,67,.15), 0 1px 2px rgba(60,64,67,.10);
    border-top: 4px solid var(--accent, #4285F4); height: 100%;
}
.kpi-label { font-size: 12px; color: #5F6368; font-weight: 500;
    text-transform: uppercase; letter-spacing: .05em; }
.kpi-value { font-family: 'Roboto Mono', monospace; font-size: 26px; font-weight: 700;
    color: #202124; margin-top: 4px; }

.section-header { display: flex; align-items: center; gap: 8px; margin: 4px 0 12px 0; }
.section-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.section-title { font-size: 16px; font-weight: 600; color: #202124; }

button[data-baseweb="tab"] { font-weight: 600; font-size: 15px; }
[data-baseweb="tab-highlight"] { background-color: #4285F4 !important; }
[data-testid="stMetricValue"] { font-family: 'Roboto Mono', monospace; }
</style>
"""


def kpi_card(label: str, value: str, color: str) -> str:
    return (f'<div class="kpi-card" style="--accent:{color}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div>')


def section_header(text: str, color: str):
    st.markdown(
        f'<div class="section-header"><span class="section-dot" style="background:{color}"></span>'
        f'<span class="section-title">{text}</span></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 420):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Roboto, sans-serif", color="#202124", size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY, height=height, margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(font=dict(family="Roboto Mono, monospace", size=12),
                        bgcolor="white", bordercolor="#DADCE0"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#DADCE0")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F3F4", zerolinecolor="#DADCE0")
    return fig


# ----------------------------------------------------------------------
# Data loading — robust, alias-based column matching
# ----------------------------------------------------------------------

REQUIRED_CANONICAL = [
    "Start Date", "Portfolio name", "Campaign Name", "Ad Group Name",
    "Customer Search Term", "Match Type", "Impressions", "Clicks",
    "Spend", "Sales", "Orders",
]

COLUMN_ALIASES = {
    "Start Date": ["start date", "date"],
    "End Date": ["end date"],
    "Portfolio name": ["portfolio name"],
    "Campaign Name": ["campaign name"],
    "Ad Group Name": ["ad group name"],
    "Customer Search Term": ["customer search term"],
    "Match Type": ["match type"],
    "Impressions": ["impressions"],
    "Clicks": ["clicks"],
    "Spend": ["spend"],
    "Sales": ["7 day total sales", "7 day total sales ₹", "sales"],
    "ACOS_reported": ["total advertising cost of sales", "acos reported"],
    "ROAS_reported": ["total return on advertising spend", "roas reported"],
    "Orders": ["7 day total orders", "7 day total orders #", "orders"],
    "Units": ["7 day total units", "7 day total units #", "units"],
    "CPC_reported": ["cost per click", "cpc reported"],
    "CTR_reported": ["click through rate", "click thru rate", "ctr reported"],
    "CVR_reported": ["7 day conversion rate", "cvr reported"],
}


def normalize_col(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def resolve_columns(columns) -> dict:
    norm_to_orig = {}
    for c in columns:
        norm = normalize_col(c)
        norm_to_orig.setdefault(norm, c)
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in norm_to_orig and norm_to_orig[alias] not in rename:
                rename[norm_to_orig[alias]] = canonical
                break
    return rename


@st.cache_data(show_spinner=False)
def load_report(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=resolve_columns(df.columns))

    missing = [c for c in REQUIRED_CANONICAL if c not in df.columns]
    if missing:
        available = ", ".join(sorted(str(c) for c in df.columns))
        raise ValueError(
            f"Report is missing expected columns: {missing}. Columns found: {available}"
        )

    df["Start Date"] = pd.to_datetime(df["Start Date"])
    df["Portfolio name"] = df["Portfolio name"].fillna("No Portfolio")
    df["Customer Search Term"] = df["Customer Search Term"].fillna("").astype(str)
    df["Search Term (normalized)"] = df["Customer Search Term"].str.strip().str.lower()
    for col in ["Impressions", "Clicks", "Spend", "Sales", "Orders"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    mt = df["Match Type"].astype(str).str.strip().str.upper()
    df["MT Group"] = np.where(mt.isin(["EXACT", "PHRASE", "BROAD"]), mt, "AUTO")

    no_sale = df["Sales"] <= 0
    df["Waste Spend"] = np.where(no_sale, df["Spend"], 0.0)
    df["Waste Clicks"] = np.where(no_sale, df["Clicks"], 0.0)
    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["CTR"] = np.where(out["Impressions"] > 0, out["Clicks"] / out["Impressions"], np.nan)
    out["CVR"] = np.where(out["Clicks"] > 0, out["Orders"] / out["Clicks"], np.nan)
    out["CPC"] = np.where(out["Clicks"] > 0, out["Spend"] / out["Clicks"], np.nan)
    out["ACOS"] = np.where(out["Sales"] > 0, out["Spend"] / out["Sales"], np.nan)
    out["ROAS"] = np.where(out["Spend"] > 0, out["Sales"] / out["Spend"], np.nan)
    if "Waste Spend" in out.columns:
        out["Waste %"] = np.where(out["Spend"] > 0, out["Waste Spend"] / out["Spend"], np.nan)
    return out


def aggregate(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    spec = dict(
        Impressions=("Impressions", "sum"), Clicks=("Clicks", "sum"),
        Spend=("Spend", "sum"), Sales=("Sales", "sum"), Orders=("Orders", "sum"),
    )
    if "Waste Spend" in df.columns:
        spec["Waste Spend"] = ("Waste Spend", "sum")
    if "Waste Clicks" in df.columns:
        spec["Waste Clicks"] = ("Waste Clicks", "sum")
    agg = df.groupby(group_cols, as_index=False).agg(**spec)
    return compute_metrics(agg)


def build_cannibalization_table(df, min_campaigns, min_clicks) -> pd.DataFrame:
    grouped = aggregate(
        df, ["Search Term (normalized)", "Customer Search Term", "Campaign Name", "Ad Group Name"]
    )
    display_term = (
        df.groupby("Search Term (normalized)")["Customer Search Term"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    )
    grouped["Search Term"] = grouped["Search Term (normalized)"].map(display_term)

    combo = grouped["Campaign Name"].astype(str) + " || " + grouped["Ad Group Name"].astype(str)
    grouped["Distinct Campaign x AdGroup"] = combo.groupby(grouped["Search Term (normalized)"]).transform("nunique")

    cannib = grouped[grouped["Distinct Campaign x AdGroup"] >= min_campaigns].copy()
    if min_clicks > 0:
        keep = cannib.groupby("Search Term (normalized)")["Clicks"].transform("max") >= min_clicks
        cannib = cannib[keep]
    return cannib.sort_values(["Search Term (normalized)", "ROAS"], na_position="last")


def build_insights_tables(df, min_wasted_spend, min_harvest_orders, target_acos, harvest_max_acos):
    term_level = aggregate(
        df, ["Search Term (normalized)", "Customer Search Term", "Campaign Name", "Ad Group Name", "Match Type"]
    )
    term_totals = aggregate(df, ["Search Term (normalized)", "Customer Search Term"])
    has_exact = df[df["Match Type"].astype(str).str.upper() == "EXACT"]["Search Term (normalized)"].unique()
    term_totals["Has Exact Match"] = term_totals["Search Term (normalized)"].isin(has_exact)

    top_ctx = (term_level.sort_values("Spend", ascending=False)
               .drop_duplicates("Search Term (normalized)")
               [["Search Term (normalized)", "Campaign Name", "Ad Group Name", "Match Type"]])
    term_totals = term_totals.merge(top_ctx, on="Search Term (normalized)", how="left")
    term_totals = term_totals.rename(columns={"Customer Search Term": "Search Term"})

    wasted = term_totals[(term_totals["Sales"] == 0) & (term_totals["Spend"] >= min_wasted_spend)] \
        .sort_values("Spend", ascending=False)
    harvest = term_totals[(~term_totals["Has Exact Match"]) & (term_totals["Orders"] >= min_harvest_orders)
                          & (term_totals["Sales"] > 0) & (term_totals["ACOS"] <= harvest_max_acos / 100)] \
        .sort_values("Sales", ascending=False)
    high_acos = term_totals[(term_totals["Sales"] > 0) & (term_totals["ACOS"] > target_acos / 100)] \
        .sort_values("Spend", ascending=False)
    return wasted, harvest, high_acos


def highlight_worst_cannib(display_df: pd.DataFrame):
    styles = pd.DataFrame("", index=display_df.index, columns=display_df.columns)
    for _term, g in display_df.groupby("Search Term"):
        no_sale = g[(g["Sales"] <= 0) & (g["Spend"] > 0)]
        if len(no_sale):
            worst = no_sale["Spend"].idxmax()
        elif g["ACOS %"].notna().any():
            worst = g["ACOS %"].idxmax()
        else:
            worst = g["Spend"].idxmax()
        styles.loc[worst, :] = "background-color: #ffc2c2; color: #7a0000; font-weight: 600;"
    return styles


ALL_PORT = "— All portfolios —"
ALL_CAMP = "— All campaigns —"
ALL_MT = "— All match types —"

EXP_GREEN = "#D6F5DE"
EXP_RED = "#FBD8D8"
EXP_GREY = "#F1F3F4"

EXPLORER_LEAF = [
    ("Search Term", "Customer Search Term", "text"),
    ("Impr", "Impressions", "int"), ("Clicks", "Clicks", "int"),
    ("CTR", "CTR", "pct"), ("Spend", "Spend", "money"), ("CPC", "CPC", "money"),
    ("Sales", "Sales", "money"), ("ACOS", "ACOS", "pct"),
]
MATCHTYPE_LEAF = [
    ("Search Term", "Customer Search Term", "text"), ("Match Type", "MT Group", "text"),
    ("Impr", "Impressions", "int"), ("Clicks", "Clicks", "int"),
    ("W.Clicks", "Waste Clicks", "int"), ("Spend", "Spend", "money"),
    ("W.Spend", "Waste Spend", "money"), ("Sales", "Sales", "money"), ("ACOS", "ACOS", "pct"),
]
WASTECLICK_LEAF = [
    ("Search Term", "Customer Search Term", "text"),
    ("Clicks", "Clicks", "int"), ("W.Clicks", "Waste Clicks", "int"),
    ("Spend", "Spend", "money"), ("W.Spend", "Waste Spend", "money"),
    ("Sales", "Sales", "money"), ("Orders", "Orders", "int"), ("ACOS", "ACOS", "pct"),
]
FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700'
    '&family=Roboto+Mono:wght@500;700&display=swap" rel="stylesheet">'
)

EXPLORER_CSS_INNER = """
html, body { margin:0; padding:0; background:transparent; font-family:'Roboto',sans-serif; }
.explorer-tree details { border:1px solid #E0E3E7; border-radius:10px; margin:6px 0; overflow:hidden; background:#fff; }
.explorer-tree details details { margin:6px 8px; }
.explorer-tree summary { list-style:none; cursor:pointer; display:flex; align-items:center; gap:10px;
    padding:10px 14px; font-family:'Roboto',sans-serif; font-weight:600; color:#202124; }
.explorer-tree summary::-webkit-details-marker { display:none; }
.explorer-tree summary::before { content:"\\25B8"; color:#5F6368; font-size:12px; }
.explorer-tree details[open] > summary::before { content:"\\25BE"; }
.explorer-tree .sum-port   { background:#E8F0FE; }
.explorer-tree .sum-camp   { background:#F1F3F4; }
.explorer-tree .sum-ag     { background:#FEF7E0; }
.explorer-tree .node-name { flex:1 1 auto; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.explorer-tree .pills { display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }
.explorer-tree .pill { font-family:'Roboto Mono',monospace; font-size:11px; font-weight:500; color:#3C4043;
    background:#fff; border:1px solid #DADCE0; border-radius:999px; padding:2px 8px; white-space:nowrap; }
.explorer-tree .pill.waste { color:#B3261E; border-color:#F3B4AE; background:#FEF3F2; }
.explorer-tree .pill b { color:#5F6368; font-weight:700; }
.explorer-tree .term-wrap { padding:6px 14px 12px 14px; overflow-x:auto; }
.explorer-tree table.term-table { width:100%; border-collapse:collapse; font-family:'Roboto',sans-serif; font-size:12.5px; }
.explorer-tree table.term-table th { text-align:right; padding:6px 10px; color:#5F6368; font-weight:600;
    border-bottom:2px solid #DADCE0; background:#fff; font-size:11px; text-transform:uppercase; letter-spacing:.03em;
    cursor:pointer; user-select:none; white-space:nowrap; }
.explorer-tree table.term-table th:hover { color:#202124; }
.explorer-tree table.term-table th.txt { text-align:left; }
.explorer-tree table.term-table th .arrow { color:#4285F4; font-size:10px; margin-left:3px; }
.explorer-tree table.term-table td { padding:6px 10px; border-bottom:1px solid #EEF0F2; }
.explorer-tree table.term-table td.num { text-align:right; font-family:'Roboto Mono',monospace; white-space:nowrap; }
.explorer-tree table.term-table td.txtcell { text-align:left; max-width:460px; }
.explorer-tree .badge { display:inline-block; padding:2px 9px; border-radius:6px; font-weight:700;
    color:#202124; border:1px solid rgba(0,0,0,.08); min-width:44px; text-align:center; }
"""

TREE_SORT_JS = """
document.querySelectorAll('table.term-table').forEach(function(tbl){
  var state = {};
  var ths = tbl.querySelectorAll('thead th');
  ths.forEach(function(th, ci){
    th.addEventListener('click', function(){
      var tbody = tbl.querySelector('tbody');
      var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
      var asc = !(state[ci] === true); state[ci] = asc;
      rows.sort(function(a, b){
        var ca = a.children[ci], cb = b.children[ci];
        var sa = ca.getAttribute('data-sort'), sb = cb.getAttribute('data-sort');
        if (sa !== null || sb !== null) {
          var va = (sa === null || sa === '') ? null : parseFloat(sa);
          var vb = (sb === null || sb === '') ? null : parseFloat(sb);
          if (va === null && vb === null) return 0;
          if (va === null) return 1;
          if (vb === null) return -1;
          return asc ? va - vb : vb - va;
        }
        var ta = ca.textContent.trim().toLowerCase();
        var tb2 = cb.textContent.trim().toLowerCase();
        if (ta < tb2) return asc ? -1 : 1;
        if (ta > tb2) return asc ? 1 : -1;
        return 0;
      });
      ths.forEach(function(h){ var a = h.querySelector('.arrow'); if (a) a.remove(); });
      var arrow = document.createElement('span');
      arrow.className = 'arrow'; arrow.textContent = asc ? '\\u25B2' : '\\u25BC';
      th.appendChild(arrow);
      rows.forEach(function(r){ tbody.appendChild(r); });
    });
  });
});
"""


def _esc(s) -> str:
    return html.escape("" if s is None else str(s))


def _fmt_money(x) -> str:
    return "—" if pd.isna(x) else f"₹{x:,.2f}"


def _fmt_num(x) -> str:
    return "—" if pd.isna(x) else f"{x:,.0f}"


def _fmt_pct(x) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.1f}%"


def _row_color(sales, spend, acos, thr_frac) -> str:
    if spend is None or spend <= 0:
        return EXP_GREY
    if sales is None or sales <= 0:
        return EXP_RED
    if pd.notna(acos) and acos < thr_frac:
        return EXP_GREEN
    return EXP_RED


def _pill(label, value, waste=False) -> str:
    cls = "pill waste" if waste else "pill"
    return f'<span class="{cls}"><b>{label}</b> {value}</span>'


def _summary_line(name, sub: pd.DataFrame, level: str, show_clicks_waste: bool = False) -> str:
    clk = sub["Clicks"].sum()
    spend = sub["Spend"].sum()
    sales = sub["Sales"].sum()
    waste = sub["Waste Spend"].sum() if "Waste Spend" in sub.columns else np.nan
    wclk = sub["Waste Clicks"].sum() if "Waste Clicks" in sub.columns else np.nan
    acos = spend / sales if sales > 0 else np.nan
    icon = {"port": "📁", "camp": "📣", "ag": "🎯"}[level]
    parts = [_pill("Clicks", _fmt_num(clk))]
    if show_clicks_waste:
        parts.append(_pill("W.Clicks", _fmt_num(wclk), waste=True))
    parts += [
        _pill("Spend", _fmt_money(spend)),
        _pill("Waste", _fmt_money(waste), waste=True),
        _pill("Sales", _fmt_money(sales)),
        _pill("ACOS", _fmt_pct(acos)),
    ]
    pills = "".join(parts)
    return (f'<summary class="sum-{level}">'
            f'<span class="node-name">{icon} {_esc(name)}</span>'
            f'<span class="pills">{pills}</span></summary>')


def _cell(kind, v):
    if kind == "int":
        return _fmt_num(v), ("" if pd.isna(v) else f"{v:.0f}")
    if kind == "money":
        return _fmt_money(v), ("" if pd.isna(v) else f"{v:.6f}")
    if kind == "pct":
        return _fmt_pct(v), ("" if pd.isna(v) else f"{float(v):.8f}")
    return _esc(v), ""


SORT_METRICS = ["Spend", "Sales", "Waste Spend", "Waste Clicks", "ACOS"]


def _group_metric(gdf: pd.DataFrame, metric: str) -> float:
    if metric == "Sales":
        return float(gdf["Sales"].sum())
    if metric == "Waste Spend":
        return float(gdf["Waste Spend"].sum()) if "Waste Spend" in gdf.columns else 0.0
    if metric == "Waste Clicks":
        return float(gdf["Waste Clicks"].sum()) if "Waste Clicks" in gdf.columns else 0.0
    if metric == "ACOS":
        spend, sales = float(gdf["Spend"].sum()), float(gdf["Sales"].sum())
        return spend / sales if sales > 0 else float("inf")
    return float(gdf["Spend"].sum())


def _ordered_keys(frame: pd.DataFrame, col: str, metric: str) -> list:
    scored = [(k, _group_metric(g, metric)) for k, g in frame.groupby(col, sort=False)]
    scored.sort(key=lambda t: (-1e18 if pd.isna(t[1]) else t[1]), reverse=True)
    return [k for k, _ in scored]


def _sort_leaf(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "ACOS":
        s = frame["ACOS"].fillna(np.inf) if "ACOS" in frame.columns else 0
        return frame.assign(__s=s).sort_values("__s", ascending=False).drop(columns="__s")
    col = metric if metric in frame.columns else "Spend"
    return frame.sort_values(col, ascending=False)


def _term_table(adf: pd.DataFrame, thr_frac: float, leaf_cols, cell_mode="row", sort_by="Spend") -> str:
    rows = _sort_leaf(adf, sort_by)
    ths = []
    for header, _key, kind in leaf_cols:
        cls = "txt" if kind == "text" else "num"
        ths.append(f'<th class="{cls}">{_esc(header)}</th>')
    head = "<tr>" + "".join(ths) + "</tr>"

    trs = []
    for _, r in rows.iterrows():
        color = _row_color(r.get("Sales"), r.get("Spend"), r.get("ACOS"), thr_frac)
        row_bg = color if cell_mode == "row" else "#ffffff"
        tds = []
        for _header, key, kind in leaf_cols:
            v = r[key]
            if kind == "text":
                disp = _esc(v)
                if key == "Customer Search Term" and (v is None or str(v).strip() == ""):
                    disp = "(blank)"
                tds.append(f'<td class="txtcell">{disp}</td>')
            else:
                disp, ds = _cell(kind, v)
                if cell_mode == "badge" and key == "ACOS":
                    tds.append(f'<td class="num" data-sort="{ds}">'
                               f'<span class="badge" style="background:{color}">{disp}</span></td>')
                else:
                    tds.append(f'<td class="num" data-sort="{ds}">{disp}</td>')
        trs.append(f'<tr style="background:{row_bg}">' + "".join(tds) + '</tr>')
    return (f'<div class="term-wrap"><table class="term-table">'
            f'<thead>{head}</thead><tbody>{"".join(trs)}</tbody></table></div>')


def _prep_names(df_scope: pd.DataFrame) -> pd.DataFrame:
    d = df_scope.copy()
    for c in ["Portfolio name", "Campaign Name", "Ad Group Name"]:
        if c in d.columns:
            d[c] = d[c].fillna("(none)").replace("", "(none)")
    return d


def build_tree_html(terms, thr_frac, open_port, open_camp, open_ag, leaf_cols,
                    cell_mode="row", sort_by="Spend", show_clicks_waste=False):
    p = " open" if open_port else ""
    c = " open" if open_camp else ""
    a = " open" if open_ag else ""
    parts = ['<div class="explorer-tree">']
    for portfolio in _ordered_keys(terms, "Portfolio name", sort_by):
        pdf = terms[terms["Portfolio name"] == portfolio]
        parts.append(f'<details{p}>')
        parts.append(_summary_line(portfolio, pdf, "port", show_clicks_waste))
        for campaign in _ordered_keys(pdf, "Campaign Name", sort_by):
            cdf = pdf[pdf["Campaign Name"] == campaign]
            parts.append(f'<details{c}>')
            parts.append(_summary_line(campaign, cdf, "camp", show_clicks_waste))
            for adgroup in _ordered_keys(cdf, "Ad Group Name", sort_by):
                adf = cdf[cdf["Ad Group Name"] == adgroup]
                parts.append(f'<details{a}>')
                parts.append(_summary_line(adgroup, adf, "ag", show_clicks_waste))
                parts.append(_term_table(adf, thr_frac, leaf_cols, cell_mode, sort_by))
                parts.append('</details>')
            parts.append('</details>')
        parts.append('</details>')
    parts.append('</div>')
    return "".join(parts)


def _render_doc(inner_html, height):
    doc = FONT_LINK + "<style>" + EXPLORER_CSS_INNER + "</style>" + inner_html \
        + "<script>" + TREE_SORT_JS + "</script>"
    components.html(doc, height=height, scrolling=True)


def _estimate_tree_height(terms, open_port, open_camp, open_ag):
    HDR, ROW = 46, 33
    n_camp = terms[["Portfolio name", "Campaign Name"]].drop_duplicates().shape[0]
    n_ag = terms[["Portfolio name", "Campaign Name", "Ad Group Name"]].drop_duplicates().shape[0]
    n_leaf = len(terms)
    h = terms["Portfolio name"].nunique() * HDR + 24
    if open_port:
        h += n_camp * HDR
    if open_camp:
        h += n_ag * HDR
    if open_ag:
        h += n_ag * HDR + n_leaf * ROW
    return int(min(max(h, 220), 850))


def render_tree(terms, thr_frac, open_port, open_camp, open_ag, leaf_cols,
                cell_mode="row", sort_by="Spend", show_clicks_waste=False):
    inner = build_tree_html(terms, thr_frac, open_port, open_camp, open_ag, leaf_cols,
                            cell_mode, sort_by, show_clicks_waste)
    _render_doc(inner, _estimate_tree_height(terms, open_port, open_camp, open_ag))


def spend_efficiency_pie(frame: pd.DataFrame, height: int = 340):
    spend = float(frame["Spend"].sum())
    waste = float(frame["Waste Spend"].sum()) if "Waste Spend" in frame.columns else 0.0
    converting = max(spend - waste, 0.0)
    sales = float(frame["Sales"].sum())
    acos = spend / sales if sales > 0 else float("nan")
    fig = px.pie(names=["Converting Spend", "Waste Spend", "Ad Sales"],
                 values=[converting, waste, sales], hole=0.58,
                 color_discrete_sequence=["#34A853", "#EA4335", SALES_TEAL])
    fig.update_traces(sort=False, direction="clockwise",
                      hovertemplate="%{label}<br>₹%{value:,.2f}<br>%{percent}<extra></extra>")
    style_fig(fig, height=height)
    acos_txt = f"ACOS<br><b>{acos*100:.1f}%</b>" if pd.notna(acos) else "ACOS<br><b>—</b>"
    fig.update_layout(annotations=[dict(text=acos_txt, x=0.5, y=0.5, showarrow=False,
                                        font=dict(size=15, family="Roboto", color="#202124"))],
                      legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
    return fig


WASTE_CLICK_BUCKETS = [
    ("1 click", 1, 1),
    ("2 clicks", 2, 2),
    ("3 clicks", 3, 3),
    ("4 clicks", 4, 4),
    ("5 clicks", 5, 5),
    ("5-10 clicks", 6, 10),
    ("10-50 clicks", 11, 50),
    ("50+ clicks", 51, np.inf),
]
WASTE_BUCKET_ORDER = [b[0] for b in WASTE_CLICK_BUCKETS]


def _bucket_label(clicks: float) -> str:
    for label, lo, hi in WASTE_CLICK_BUCKETS:
        if lo <= clicks <= hi:
            return label
    return WASTE_CLICK_BUCKETS[-1][0]


def build_waste_click_terms(df: pd.DataFrame) -> pd.DataFrame:
    grouped = aggregate(
        df, ["Portfolio name", "Campaign Name", "Ad Group Name", "Customer Search Term"])
    pure_waste = grouped[(grouped["Sales"] <= 0) & (grouped["Clicks"] > 0)].copy()
    pure_waste["Bucket"] = pure_waste["Clicks"].apply(_bucket_label)
    return pure_waste


def waste_click_bucket_summary(waste_terms: pd.DataFrame) -> pd.DataFrame:
    if waste_terms.empty:
        return pd.DataFrame(columns=["Bucket", "Search Terms", "Waste Clicks", "Waste Spend"])
    summ = (waste_terms.groupby("Bucket")
            .agg(**{"Search Terms": ("Customer Search Term", "count"),
                    "Waste Clicks": ("Clicks", "sum"),
                    "Waste Spend": ("Spend", "sum")})
            .reindex(WASTE_BUCKET_ORDER).dropna(how="all").reset_index())
    summ["Search Terms"] = summ["Search Terms"].fillna(0).astype(int)
    summ["Waste Clicks"] = summ["Waste Clicks"].fillna(0)
    summ["Waste Spend"] = summ["Waste Spend"].fillna(0)
    return summ


def color_legend_html(thr_pct: float, badge=False) -> str:
    def sw(c):
        return (f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;'
                f'vertical-align:middle;margin-right:6px;background:{c};border:1px solid rgba(0,0,0,.1)"></span>')
    what = "ACOS badge" if badge else "Rows"
    return (
        '<div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap;'
        'font-family:Roboto,sans-serif;font-size:12.5px;color:#5F6368;margin:2px 0 10px;">'
        f'<span>{sw(EXP_GREEN)}ACOS &lt; {thr_pct:g}%</span>'
        f'<span>{sw(EXP_RED)}ACOS ≥ {thr_pct:g}% or spend with no sales</span>'
        f'<span>{sw(EXP_GREY)}no spend</span>'
        f'<span style="margin-left:auto;color:#80868B;">{what} colored by efficiency · click a header to sort</span>'
        '</div>'
    )


# ======================================================================
# UI
# ======================================================================

def main():
    st.set_page_config(page_title="Search Term Intelligence Tool", page_icon="🔎", layout="wide")
    st.markdown(FONT_LINK, unsafe_allow_html=True)
    st.markdown(CSS, unsafe_allow_html=True)

    st.sidebar.title("🔎 Search Term Intelligence")
    uploaded = st.sidebar.file_uploader("Upload Search Term Report (.xlsx)", type=["xlsx"])

    if uploaded is None:
        st.title("Amazon Sponsored Products — Search Term Intelligence Tool")
        st.info("👈 Upload a Sponsored Products Search Term report (.xlsx) to begin. Expected columns include "
                "Start Date, Portfolio name, Campaign Name, Ad Group Name, Customer Search Term, Match Type, "
                "Impressions, Clicks, Spend, 7 Day Total Sales, 7 Day Total Orders.")
        st.stop()

    try:
        raw = load_report(uploaded)
    except Exception as e:
        st.error(f"Couldn't read this file: {e}")
        st.stop()

    st.sidebar.markdown("---")
    fba_only = st.sidebar.checkbox(
        "Only FBA portfolios (excludes Vizari)", value=True,
        help="Keeps any portfolio whose name contains 'FBA', but always excludes portfolios containing 'Vizari'.")
    is_fba = raw["Portfolio name"].str.contains("FBA", case=False, na=False)
    is_vizari = raw["Portfolio name"].str.contains("Vizari", case=False, na=False)
    scope_mask = (is_fba & ~is_vizari) if fba_only else pd.Series(True, index=raw.index)
    raw_scoped = raw[scope_mask]

    if fba_only:
        overlap = raw["Portfolio name"][is_fba & is_vizari].nunique()
        st.sidebar.caption(
            f"Scoped to {raw_scoped['Portfolio name'].nunique()} FBA portfolios. "
            f"{overlap} portfolio(s) excluded for containing both 'FBA' and 'Vizari'.")

    if raw_scoped.empty:
        st.error("No rows match an 'FBA' portfolio name (excluding Vizari). Uncheck the filter to see all data.")
        st.stop()

    min_date, max_date = raw_scoped["Start Date"].min().date(), raw_scoped["Start Date"].max().date()
    date_range = st.sidebar.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    portfolios = sorted(raw_scoped["Portfolio name"].dropna().unique())
    selected_portfolios = st.sidebar.multiselect("Portfolio", portfolios, default=[])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Cannibalization settings")
    min_campaigns = st.sidebar.number_input(
        "Min distinct Campaign × Ad Group combos to flag", min_value=2, value=2, step=1)
    min_clicks_cannib = st.sidebar.number_input(
        "Min clicks (at least one instance) to include", min_value=0, value=1, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Insights settings")
    min_wasted_spend = st.sidebar.number_input(
        "Min spend (₹) to flag a zero-sale term", min_value=0.0, value=5.0, step=1.0)
    min_harvest_orders = st.sidebar.number_input(
        "Min orders to flag a harvest opportunity", min_value=1, value=2, step=1)
    target_acos_pct = st.sidebar.number_input(
        "Target ACOS % (terms above get flagged)", min_value=1.0, value=30.0, step=1.0)

    df = raw_scoped[(raw_scoped["Start Date"].dt.date >= start_date) & (raw_scoped["Start Date"].dt.date <= end_date)].copy()
    if selected_portfolios:
        df = df[df["Portfolio name"].isin(selected_portfolios)]
    if df.empty:
        st.warning("No rows match the current filters.")
        st.stop()

    st.title("Amazon Sponsored Products — Search Term Intelligence Tool")
    scope_label = "FBA portfolios only (excl. Vizari)" if fba_only else "all portfolios"
    st.caption(f"{df['Campaign Name'].nunique()} campaigns · {df['Customer Search Term'].nunique()} search terms "
               f"· {scope_label} · {start_date} → {end_date} · {len(df):,} rows")

    (tab_dashboard, tab_cannibalization, tab_insights,
     tab_explorer, tab_matchtype, tab_wasteclicks) = st.tabs(
        ["📊 Dashboard", "🚨 Keyword Cannibalization",
         "💡 Insights & Recommendations", "🗂️ Portfolio Explorer",
         "🎯 Campaigns × Match Type", "🖱️ Waste Clicks"])

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    with tab_dashboard:
        totals = compute_metrics(pd.DataFrame([{
            "Impressions": df["Impressions"].sum(), "Clicks": df["Clicks"].sum(),
            "Spend": df["Spend"].sum(), "Sales": df["Sales"].sum(), "Orders": df["Orders"].sum(),
            "Waste Spend": df["Waste Spend"].sum(), "Waste Clicks": df["Waste Clicks"].sum(),
        }])).iloc[0]

        kpis = [
            ("Ad Sales", f"₹{totals['Sales']:,.0f}", PALETTE["green"]),
            ("Spend", f"₹{totals['Spend']:,.0f}", PALETTE["blue"]),
            ("Waste Spend", f"₹{totals['Waste Spend']:,.0f}", PALETTE["red"]),
            ("ROAS", f"{totals['ROAS']:.2f}" if pd.notna(totals['ROAS']) else "—", PALETTE["purple"]),
            ("ACOS", f"{totals['ACOS']*100:.1f}%" if pd.notna(totals['ACOS']) else "—", PALETTE["yellow"]),
            ("CTR", f"{totals['CTR']*100:.2f}%" if pd.notna(totals['CTR']) else "—", PALETTE["teal"]),
        ]
        for col, (label, value, color) in zip(st.columns(6), kpis):
            with col:
                st.markdown(kpi_card(label, value, color), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        daily = aggregate(df, ["Start Date"]).sort_values("Start Date")

        def spend_or_sales_vs_acos(value_col, bar_name, bar_hex):
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(
                x=daily["Start Date"], y=daily[value_col], name=bar_name,
                marker=dict(color=daily[value_col], colorscale=BLUE_SCALE, showscale=False,
                            line=dict(width=0.5, color="white")),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>" + bar_name + ": ₹%{y:,.2f}<extra></extra>",
            )
            fig.add_trace(go.Scatter(
                x=daily["Start Date"], y=daily["ACOS"] * 100, name="ACOS %", mode="lines+markers",
                line=dict(color=ACOS_LINE, width=2.5), marker=dict(size=5),
                hovertemplate="ACOS: %{y:.1f}%<extra></extra>",
            ), secondary_y=True)
            fig.update_yaxes(title_text=f"{bar_name} (₹)", secondary_y=False)
            fig.update_yaxes(title_text="ACOS %", secondary_y=True, showgrid=False, ticksuffix="%")
            style_fig(fig, height=380)
            fig.update_layout(hovermode="x unified", bargap=0.25,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_xaxes(tickformat="%b %d")
            return fig

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                section_header("Spend vs ACOS by Date", PALETTE["blue"])
                st.plotly_chart(spend_or_sales_vs_acos("Spend", "Spend", SPEND_BLUE), use_container_width=True, key="dash_spend_acos")
        with c2:
            with st.container(border=True):
                section_header("Sales vs ACOS by Date", PALETTE["blue"])
                st.plotly_chart(spend_or_sales_vs_acos("Sales", "Ad Sales", SPEND_BLUE), use_container_width=True, key="dash_sales_acos")

        c3, c4 = st.columns(2)
        by_match = aggregate(df, ["MT Group"]).sort_values("Spend", ascending=False)
        with c3:
            with st.container(border=True):
                section_header("Waste Spend by Match Type", PALETTE["red"])
                fig = px.pie(by_match, names="MT Group", values="Waste Spend", hole=0.45,
                             color_discrete_sequence=COLORWAY)
                fig.update_traces(hovertemplate="%{label}<br>Waste Spend: ₹%{value:,.2f}<br>%{percent}<extra></extra>")
                style_fig(fig, height=340)
                st.plotly_chart(fig, use_container_width=True, key="dash_waste_by_mt")
        with c4:
            with st.container(border=True):
                section_header("Sales by Match Type", PALETTE["blue"])
                fig = px.pie(by_match, names="MT Group", values="Sales", hole=0.45,
                             color_discrete_sequence=COLORWAY)
                fig.update_traces(hovertemplate="%{label}<br>Sales: ₹%{value:,.2f}<br>%{percent}<extra></extra>")
                style_fig(fig, height=340)
                st.plotly_chart(fig, use_container_width=True, key="dash_sales_by_mt")

        with st.container(border=True):
            section_header("Portfolio Breakdown", PALETTE["yellow"])
            by_port = aggregate(df, ["Portfolio name"]).sort_values("Spend", ascending=False)
            disp = by_port.copy()
            disp["ACOS"] = (disp["ACOS"] * 100).round(1)
            disp["ROAS"] = disp["ROAS"].round(2)
            disp["CTR"] = (disp["CTR"] * 100).round(2)
            disp = disp[["Portfolio name", "Impressions", "Clicks", "CTR", "Spend",
                         "Waste Spend", "Sales", "Orders", "ACOS", "ROAS"]]
            st.dataframe(disp.rename(columns={"CTR": "CTR %", "ACOS": "ACOS %"}),
                         use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Keyword Cannibalization
    # ------------------------------------------------------------------
    with tab_cannibalization:
        section_header("Search terms triggering multiple Campaign × Ad Group combinations", PALETTE["red"])
        cannib = build_cannibalization_table(df, min_campaigns, min_clicks_cannib)

        if cannib.empty:
            st.success("No cannibalized search terms found at the current thresholds/filters.")
        else:
            fcol1, fcol2 = st.columns([2, 1.4])
            with fcol1:
                search_filter = st.text_input("🔍 Filter by search term")
            with fcol2:
                rank_by = st.selectbox("Rank term-groups by", SORT_METRICS, index=0, key="cannib_rank")

            show = cannib.copy()
            if search_filter:
                show = show[show["Search Term"].str.contains(search_filter, case=False, na=False)]

            if show.empty:
                st.info("No cannibalized terms match this search filter.")
            else:
                n_terms = show["Search Term (normalized)"].nunique()
                total_waste_spend = show["Waste Spend"].sum() if "Waste Spend" in show.columns else 0.0
                total_waste_clicks = show["Waste Clicks"].sum() if "Waste Clicks" in show.columns else 0.0
                total_sales = show["Sales"].sum()
                total_spend = show["Spend"].sum()
                overall_acos = (total_spend / total_sales) if total_sales > 0 else np.nan

                overview = [
                    ("Cannibalized search terms", f"{n_terms:,}", PALETTE["red"]),
                    ("Waste clicks (no-sale rows)", f"{total_waste_clicks:,.0f}", PALETTE["yellow"]),
                    ("Waste spend (no-sale rows)", f"₹{total_waste_spend:,.2f}", PALETTE["red"]),
                    ("Sales (all cannibalized terms)", f"₹{total_sales:,.2f}", PALETTE["green"]),
                    ("ACOS (cannibalized overall)",
                     f"{overall_acos*100:.1f}%" if pd.notna(overall_acos) else "—", PALETTE["purple"]),
                ]
                for col, (label, value, color) in zip(st.columns(5), overview):
                    with col:
                        st.markdown(kpi_card(label, value, color), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                g = show.groupby("Search Term")
                if rank_by == "ACOS":
                    gscore = (g["Spend"].transform("sum") /
                              g["Sales"].transform("sum").replace(0, np.nan)).fillna(np.inf)
                elif rank_by == "Waste Clicks":
                    gscore = g["Waste Clicks"].transform("sum")
                else:
                    gscore = g[rank_by].transform("sum")
                show = show.assign(__g=gscore).sort_values(
                    ["__g", "Search Term", "ACOS"], ascending=[False, True, False], na_position="first"
                ).drop(columns="__g")

                display = show[["Search Term", "Campaign Name", "Ad Group Name",
                                "Clicks", "Waste Clicks", "Spend", "Waste Spend", "Sales", "ACOS"]].copy()
                display["ACOS %"] = (display["ACOS"] * 100).round(1)
                display[["Spend", "Waste Spend", "Sales"]] = display[["Spend", "Waste Spend", "Sales"]].round(2)
                display = display.drop(columns="ACOS").reset_index(drop=True)

                with st.container(border=True):
                    styled = display.style.apply(highlight_worst_cannib, axis=None).format(precision=2)
                    st.dataframe(styled, use_container_width=True, height=600)

                with st.container(border=True):
                    section_header("Export", PALETTE["green"])
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as w:
                        display.to_excel(w, sheet_name="Cannibalized Search Terms", index=False)
                    st.download_button("Download cannibalization report (.xlsx)", data=buf.getvalue(),
                                       file_name="keyword_cannibalization_report.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_cannib")

    # ------------------------------------------------------------------
    # Insights & Recommendations
    # ------------------------------------------------------------------
    with tab_insights:
        ins_ctrl = st.columns([2, 3, 3])
        with ins_ctrl[0]:
            ins_mt_opts = [ALL_MT] + sorted(df["MT Group"].dropna().unique())
            ins_sel_mt = st.selectbox("Match type", ins_mt_opts, index=0, key="ins_mt")
        with ins_ctrl[1]:
            harvest_max_acos = st.number_input(
                "Harvest cutoff — max ACOS % to qualify as a winner worth promoting to Exact",
                min_value=1.0, value=25.0, step=1.0, key="harvest_acos")
        with ins_ctrl[2]:
            high_acos_thr = st.number_input(
                "High-ACOS threshold — flag converting terms above this ACOS %",
                min_value=1.0, value=float(target_acos_pct), step=1.0, key="high_acos_thr")

        ins_df = df if ins_sel_mt == ALL_MT else df[df["MT Group"] == ins_sel_mt]
        if ins_df.empty:
            st.warning("No rows for this match type.")
        else:
            wasted, harvest, high_acos = build_insights_tables(
                ins_df, min_wasted_spend, min_harvest_orders, high_acos_thr, harvest_max_acos)
            total_spend = ins_df["Spend"].sum()
            wasted_total = wasted["Spend"].sum()
            wasted_pct = (wasted_total / total_spend * 100) if total_spend > 0 else 0

            for col, (label, value, color) in zip(st.columns(4), [
                ("Wasted spend (zero sales)", f"₹{wasted_total:,.0f}", PALETTE["red"]),
                ("% of total spend wasted", f"{wasted_pct:.1f}%", PALETTE["yellow"]),
                ("Harvest opportunities", f"{len(harvest):,}", PALETTE["green"]),
                ("High-ACOS search terms", f"{len(high_acos):,}", PALETTE["purple"]),
            ]):
                with col:
                    st.markdown(kpi_card(label, value, color), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            with st.container(border=True):
                section_header("🚫 Spend With No Sales — negative keyword candidates", PALETTE["red"])
                if wasted.empty:
                    st.success("No search terms are burning spend with zero sales at the current threshold.")
                else:
                    dw = wasted[["Search Term", "Campaign Name", "Ad Group Name", "Match Type",
                                 "Clicks", "Spend", "Impressions"]].copy()
                    dw["Spend"] = dw["Spend"].round(2)
                    st.dataframe(dw, use_container_width=True, hide_index=True, height=320)

            with st.container(border=True):
                section_header(f"🌱 Harvest Opportunities — winners under {harvest_max_acos:.0f}% ACOS to promote to Exact",
                               PALETTE["green"])
                if harvest.empty:
                    st.info("No harvest candidates at the current order threshold and ACOS cutoff.")
                else:
                    dh = harvest[["Search Term", "Campaign Name", "Ad Group Name", "Match Type",
                                  "Clicks", "Orders", "Spend", "Sales", "ACOS", "ROAS"]].copy()
                    dh["ACOS"] = (dh["ACOS"] * 100).round(1)
                    dh["ROAS"] = dh["ROAS"].round(2)
                    dh[["Spend", "Sales"]] = dh[["Spend", "Sales"]].round(2)
                    st.dataframe(dh.rename(columns={"ACOS": "ACOS %"}), use_container_width=True, hide_index=True, height=320)

            with st.container(border=True):
                section_header(f"⚠️ High ACOS Search Terms — above {high_acos_thr:.0f}% threshold", PALETTE["purple"])
                if high_acos.empty:
                    st.success(f"No converting search terms exceed {high_acos_thr:.0f}% ACOS.")
                else:
                    dha = high_acos[["Search Term", "Campaign Name", "Ad Group Name", "Match Type",
                                     "Clicks", "Orders", "Spend", "Sales", "ACOS", "ROAS"]].copy()
                    dha["ACOS"] = (dha["ACOS"] * 100).round(1)
                    dha["ROAS"] = dha["ROAS"].round(2)
                    dha[["Spend", "Sales"]] = dha[["Spend", "Sales"]].round(2)
                    st.dataframe(dha.rename(columns={"ACOS": "ACOS %"}), use_container_width=True, hide_index=True, height=320)

            with st.container(border=True):
                section_header("🥧 Converting Spend vs Waste Spend vs Ad Sales", PALETTE["teal"])
                pie_col, note_col = st.columns([1.3, 1])
                with pie_col:
                    st.plotly_chart(spend_efficiency_pie(ins_df, height=360), use_container_width=True, key="insights_eff_pie")
                with note_col:
                    tot_spend = float(ins_df["Spend"].sum())
                    tot_waste = float(ins_df["Waste Spend"].sum())
                    tot_sales = float(ins_df["Sales"].sum())
                    for label, value, color in [
                        ("Converting spend", f"₹{tot_spend - tot_waste:,.0f}", PALETTE["green"]),
                        ("Waste spend", f"₹{tot_waste:,.0f}", PALETTE["red"]),
                        ("Ad sales", f"₹{tot_sales:,.0f}", PALETTE["blue"]),
                        ("Waste share of spend", f"{(tot_waste / tot_spend * 100) if tot_spend else 0:.1f}%",
                         PALETTE["yellow"]),
                    ]:
                        st.markdown(kpi_card(label, value, color), unsafe_allow_html=True)
                        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Portfolio Explorer
    # ------------------------------------------------------------------
    with tab_explorer:
        section_header("Portfolio → Campaign → Ad Group → Search Term drill-down", PALETTE["blue"])
        ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([2, 2, 2, 1.4, 1.2])
        with ctrl1:
            port_opts = [ALL_PORT] + sorted(df["Portfolio name"].dropna().unique())
            sel_port = st.selectbox("Portfolio", port_opts, index=0, key="exp_port")
        specific_portfolio = sel_port != ALL_PORT
        camp_pool = df[df["Portfolio name"] == sel_port] if specific_portfolio else df

        with ctrl2:
            camp_query = st.text_input("Campaign search (contains)", key="exp_prefix")
        matching = sorted([c for c in camp_pool["Campaign Name"].dropna().unique()
                           if not camp_query or camp_query.lower() in str(c).lower()])
        with ctrl3:
            all_match_label = f"— All matching '{camp_query}' ({len(matching)}) —" if camp_query else ALL_CAMP
            camp_opts = [all_match_label] + matching
            if "exp_camp" in st.session_state and st.session_state["exp_camp"] not in camp_opts:
                del st.session_state["exp_camp"]
            sel_camp = st.selectbox("Campaign", camp_opts, index=0, key="exp_camp")
        with ctrl4:
            exp_sort = st.selectbox("Sort groups by", SORT_METRICS, index=0, key="exp_sort")
        with ctrl5:
            acos_thr_pct = st.number_input("ACOS green cutoff %", min_value=0.1, value=5.0, step=0.5, key="exp_acos")

        pick_all_match = sel_camp == all_match_label
        thr_frac = acos_thr_pct / 100.0

        dfx = df.copy()
        if specific_portfolio:
            dfx = dfx[dfx["Portfolio name"] == sel_port]
        if pick_all_match:
            if camp_query and matching:
                dfx = dfx[dfx["Campaign Name"].isin(matching)]
        else:
            dfx = dfx[dfx["Campaign Name"] == sel_camp]

        if dfx.empty:
            st.warning("No rows match the current selection.")
        else:
            specific_campaign = not pick_all_match
            has_match_family = pick_all_match and bool(camp_query)
            open_port = specific_portfolio or specific_campaign or has_match_family
            open_camp = specific_campaign
            open_ag = specific_campaign

            st.markdown(color_legend_html(acos_thr_pct), unsafe_allow_html=True)
            terms_exp = aggregate(_prep_names(dfx),
                                  ["Portfolio name", "Campaign Name", "Ad Group Name", "Customer Search Term"])
            render_tree(terms_exp, thr_frac, open_port, open_camp, open_ag, EXPLORER_LEAF,
                        cell_mode="row", sort_by=exp_sort)

    # ------------------------------------------------------------------
    # Campaigns × Match Type
    # ------------------------------------------------------------------
    with tab_matchtype:
        section_header("Campaign performance split by Match Type", PALETTE["purple"])
        mt_ctrl = st.columns([2, 2, 1.4, 1.2])
        with mt_ctrl[0]:
            mt_port_opts = [ALL_PORT] + sorted(df["Portfolio name"].dropna().unique())
            mt_sel_port = st.selectbox("Portfolio", mt_port_opts, index=0, key="mt_port")
        with mt_ctrl[1]:
            mt_groups = sorted(df["MT Group"].dropna().unique())
            mt_sel = st.multiselect("Match types to include", mt_groups, default=mt_groups, key="mt_types")
        with mt_ctrl[2]:
            mt_sort = st.selectbox("Sort groups by", SORT_METRICS, index=0, key="mt_sort")
        with mt_ctrl[3]:
            mt_acos_thr = st.number_input("ACOS green cutoff %", min_value=0.1, value=5.0, step=0.5, key="mt_acos")

        mt_thr = mt_acos_thr / 100.0
        dmt = df.copy()
        if mt_sel_port != ALL_PORT:
            dmt = dmt[dmt["Portfolio name"] == mt_sel_port]
        if mt_sel:
            dmt = dmt[dmt["MT Group"].isin(mt_sel)]

        if dmt.empty:
            st.warning("No rows match the current selection.")
        else:
            by_mt = aggregate(dmt, ["MT Group"]).sort_values("Spend", ascending=False)
            n_metrics = len(by_mt)
            cols = st.columns(max(n_metrics, 1))
            for col, (_, r) in zip(cols, by_mt.iterrows()):
                acos_txt = f"{r['ACOS']*100:.1f}%" if pd.notna(r["ACOS"]) else "—"
                waste = r["Waste Spend"] if "Waste Spend" in r else 0.0
                with col:
                    st.markdown(kpi_card(
                        f"{r['MT Group']} · ACOS {acos_txt}",
                        f"₹{r['Spend']:,.0f}",
                        {"EXACT": PALETTE["green"], "PHRASE": PALETTE["blue"],
                         "BROAD": PALETTE["yellow"], "AUTO": PALETTE["purple"]}.get(r["MT Group"], PALETTE["teal"]),
                    ), unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:12px;color:#B3261E;margin-top:4px'>"
                                f"Waste ₹{waste:,.0f} · Sales ₹{r['Sales']:,.0f}</div>", unsafe_allow_html=True)

            st.markdown(color_legend_html(mt_acos_thr, badge=True), unsafe_allow_html=True)
            terms_mt = aggregate(_prep_names(dmt),
                                 ["Portfolio name", "Campaign Name", "Ad Group Name", "Customer Search Term", "MT Group"])
            render_tree(terms_mt, mt_thr, open_port=(mt_sel_port != ALL_PORT), open_camp=False, open_ag=False,
                        leaf_cols=MATCHTYPE_LEAF, cell_mode="badge", sort_by=mt_sort)

    # ------------------------------------------------------------------
    # Waste Clicks
    # ------------------------------------------------------------------
    with tab_wasteclicks:
        section_header("Clicks you paid for that never converted", PALETTE["red"])
        wc_ctrl = st.columns([2, 2, 2])
        with wc_ctrl[0]:
            wc_port_opts = [ALL_PORT] + sorted(df["Portfolio name"].dropna().unique())
            wc_sel_port = st.selectbox("Portfolio", wc_port_opts, index=0, key="wc_port")
        with wc_ctrl[1]:
            wc_mt_opts = [ALL_MT] + sorted(df["MT Group"].dropna().unique())
            wc_sel_mt = st.selectbox("Match type", wc_mt_opts, index=0, key="wc_mt")
        with wc_ctrl[2]:
            wc_buckets_sel = st.multiselect("Click buckets to include", WASTE_BUCKET_ORDER,
                                            default=WASTE_BUCKET_ORDER, key="wc_buckets")

        dwc = df.copy()
        if wc_sel_port != ALL_PORT:
            dwc = dwc[dwc["Portfolio name"] == wc_sel_port]
        if wc_sel_mt != ALL_MT:
            dwc = dwc[dwc["MT Group"] == wc_sel_mt]

        waste_terms = build_waste_click_terms(_prep_names(dwc))
        if wc_buckets_sel:
            waste_terms = waste_terms[waste_terms["Bucket"].isin(wc_buckets_sel)]

        if waste_terms.empty:
            st.success("No zero-sale search terms in this scope — nothing wasted here.")
        else:
            tot_terms = len(waste_terms)
            tot_wclicks = waste_terms["Clicks"].sum()
            tot_wspend = waste_terms["Spend"].sum()
            scope_spend = dwc["Spend"].sum()
            wshare = (tot_wspend / scope_spend * 100) if scope_spend > 0 else 0
            for col, (label, value, color) in zip(st.columns(4), [
                ("Zero-sale search terms", f"{tot_terms:,}", PALETTE["red"]),
                ("Wasted clicks", f"{tot_wclicks:,.0f}", PALETTE["yellow"]),
                ("Wasted spend", f"₹{tot_wspend:,.2f}", PALETTE["red"]),
                ("Waste share of scope spend", f"{wshare:.1f}%", PALETTE["purple"]),
            ]):
                with col:
                    st.markdown(kpi_card(label, value, color), unsafe_allow_html=True)

            summ = waste_click_bucket_summary(waste_terms)
            with st.container(border=True):
                section_header("Waste by click bracket", PALETTE["yellow"])
                bar_col, pie1_col, pie2_col = st.columns([1.5, 1, 1])
                with bar_col:
                    figb = px.bar(summ, x="Bucket", y="Waste Spend",
                                  color="Waste Spend", color_continuous_scale=RED_SCALE,
                                  text="Search Terms")
                    figb.update_traces(
                        texttemplate="%{text} terms", textposition="outside",
                        hovertemplate="%{x}<br>Waste Spend: ₹%{y:,.2f}<extra></extra>")
                    style_fig(figb, height=340)
                    figb.update_layout(coloraxis_showscale=False, yaxis_title="Waste Spend (₹)",
                                       xaxis_title=None, margin=dict(t=30, b=10))
                    figb.update_xaxes(tickangle=-30)
                    st.plotly_chart(figb, use_container_width=True, key="wc_bucket_bar")
                with pie1_col:
                    figp = px.pie(summ, names="Bucket", values="Waste Clicks", hole=0.5,
                                  color_discrete_sequence=COLORWAY, category_orders={"Bucket": WASTE_BUCKET_ORDER})
                    figp.update_traces(sort=False,
                                       hovertemplate="%{label}<br>Waste Clicks: %{value:,.0f}<br>%{percent}<extra></extra>")
                    style_fig(figp, height=300)
                    figp.update_layout(title=dict(text="Wasted Clicks", x=0.5, y=0.98, font=dict(size=13)),
                                       showlegend=False)
                    st.plotly_chart(figp, use_container_width=True, key="wc_clicks_pie")
                with pie2_col:
                    figp2 = px.pie(summ, names="Bucket", values="Waste Spend", hole=0.5,
                                   color_discrete_sequence=COLORWAY, category_orders={"Bucket": WASTE_BUCKET_ORDER})
                    figp2.update_traces(sort=False,
                                        hovertemplate="%{label}<br>Waste Spend: ₹%{value:,.2f}<br>%{percent}<extra></extra>")
                    style_fig(figp2, height=300)
                    figp2.update_layout(title=dict(text="Wasted Spend", x=0.5, y=0.98, font=dict(size=13)),
                                        showlegend=False)
                    st.plotly_chart(figp2, use_container_width=True, key="wc_spend_pie")

            with st.container(border=True):
                section_header("Drill down — Portfolio → Campaign → Ad Group → wasted Search Term", PALETTE["blue"])
                wc_sort = st.selectbox("Sort groups by", ["Waste Spend", "Waste Clicks", "Spend"],
                                       index=0, key="wc_sort")
                open_p = wc_sel_port != ALL_PORT
                render_tree(waste_terms, thr_frac=0.0, open_port=open_p, open_camp=open_p, open_ag=False,
                            leaf_cols=WASTECLICK_LEAF, cell_mode="row", sort_by=wc_sort, show_clicks_waste=True)


if __name__ == "__main__":
    main()
