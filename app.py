import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import sys
import io
import builtins
from scipy.stats import (
    norm, binom, poisson, t as t_dist, f as f_dist, shapiro,
    ttest_1samp, ttest_ind, ttest_rel, f_oneway,
    chisquare, chi2_contingency, linregress
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatPortal – Inferential Statistics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a12;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #12122a 100%);
    border-right: 1px solid rgba(139, 92, 246, 0.2);
}
[data-testid="stSidebar"] * { color: #c4b5fd !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #7c3aed !important; }
[data-testid="stSidebar"] label { color: #a78bfa !important; font-size: 0.85rem !important; }

/* ── Main background ── */
.main .block-container { background: transparent; padding-top: 1.5rem; }

/* ── Hero header ── */
.hero {
    background: radial-gradient(ellipse at 20% 50%, rgba(109,40,217,0.25) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(168,85,247,0.15) 0%, transparent 60%),
                linear-gradient(135deg, #0d0d1a 0%, #12122a 50%, #0d0d1a 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    text-align: center;
    box-shadow: 0 0 60px rgba(109,40,217,0.2), inset 0 0 60px rgba(0,0,0,0.3);
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 40px,
        rgba(139,92,246,0.03) 40px, rgba(139,92,246,0.03) 41px
    ),
    repeating-linear-gradient(
        90deg, transparent, transparent 40px,
        rgba(139,92,246,0.03) 40px, rgba(139,92,246,0.03) 41px
    );
    pointer-events: none;
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #a78bfa;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    opacity: 0.8;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(135deg, #fff 0%, #c4b5fd 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    letter-spacing: -1px;
}
.hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 400;
    letter-spacing: 0.02em;
}

/* ── Landing page ── */
.landing-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 2rem 0;
}
.landing-card {
    background: rgba(109,40,217,0.06);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.landing-card:hover {
    border-color: rgba(139,92,246,0.5);
    background: rgba(109,40,217,0.12);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(109,40,217,0.2);
}
.landing-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.landing-card h4 { color: #c4b5fd; margin: 0 0 0.4rem 0; font-size: 0.95rem; font-weight: 600; }
.landing-card p { color: #64748b; font-size: 0.8rem; margin: 0; line-height: 1.4; }

.upload-box {
    background: linear-gradient(135deg, rgba(109,40,217,0.08), rgba(168,85,247,0.05));
    border: 2px dashed rgba(139,92,246,0.4);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
}
.upload-box h3 { color: #c4b5fd; margin-bottom: 0.5rem; }
.upload-box p { color: #64748b; font-size: 0.9rem; }

/* ── Metric cards ── */
.metric-card {
    background: rgba(109,40,217,0.08);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, #a855f7);
}
.metric-card:hover {
    border-color: rgba(139,92,246,0.45);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(109,40,217,0.2);
}
.metric-label { color: #7c3aed; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; font-family: 'JetBrains Mono', monospace; }
.metric-val { font-size: 2rem; font-weight: 700; margin: 4px 0 0 0; font-family: 'JetBrains Mono', monospace; }

/* ── Result / conclusion cards ── */
.result-card {
    border-left: 4px solid #7c3aed;
    border-radius: 0 12px 12px 0;
    padding: 1.25rem 1.5rem;
    margin: 1.5rem 0;
    background: rgba(109,40,217,0.07);
}
.result-card.reject   { border-left-color: #10b981; background: rgba(16,185,129,0.06); }
.result-card.fail-reject { border-left-color: #f43f5e; background: rgba(244,63,94,0.06); }

.badge {
    padding: 0.3em 0.75em;
    font-size: 0.7rem;
    font-weight: 700;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: inline-block;
    margin-bottom: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
}
.badge-sig   { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-insig { background: rgba(244,63,94,0.15);  color: #fb7185; border: 1px solid rgba(244,63,94,0.3); }

/* ── Styled tables ── */
.styled-table {
    width: 100%; border-collapse: collapse; margin: 0.75rem 0;
    font-size: 0.88rem; border-radius: 10px; overflow: hidden;
    font-family: 'JetBrains Mono', monospace;
}
.styled-table th {
    background: rgba(109,40,217,0.25);
    color: #c4b5fd; text-align: left;
    padding: 10px 14px; font-weight: 600;
    border-bottom: 1px solid rgba(139,92,246,0.3);
    letter-spacing: 0.05em; font-size: 0.8rem;
}
.styled-table td {
    padding: 9px 14px;
    border-bottom: 1px solid rgba(139,92,246,0.1);
    color: #cbd5e1;
}
.styled-table tr:hover td { background: rgba(139,92,246,0.06); }

/* ── Section headers ── */
h1, h2, h3 { color: #e2e8f0 !important; }
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(109,40,217,0.06);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(139,92,246,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8 !important;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.4rem 1rem;
    background: transparent;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    color: #fff !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4);
}

/* ── Plotly charts background ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* ── Streamlit element overrides ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stSelectbox > div > div { background: #12122a !important; border-color: rgba(139,92,246,0.3) !important; border-radius: 8px !important; color: #e2e8f0 !important; }
.stMultiSelect > div > div { background: #12122a !important; border-color: rgba(139,92,246,0.3) !important; border-radius: 8px !important; }
div[data-testid="stTextArea"] textarea {
    background: #0d0d1a !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 10px !important;
    color: #a78bfa !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stAlert { border-radius: 10px !important; }

/* ── Section divider ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #7c3aed;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    border-left: 3px solid #7c3aed;
    padding-left: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">// inferential statistics engine</div>
    <div class="hero-title">StatPortal</div>
    <div class="hero-sub">Upload → Detect → Analyse → Conclude &nbsp;·&nbsp; Automated hypothesis testing for your data</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center;padding:1rem 0 0.5rem;font-family:'JetBrains Mono',monospace;font-size:1.1rem;color:#a78bfa;letter-spacing:0.1em;">
📊 STATPORTAL
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="section-label">significance level</div>', unsafe_allow_html=True)
alpha = st.sidebar.slider("α (alpha)", 0.01, 0.10, 0.05, 0.01,
    help="Type I error rate — probability of rejecting a true H₀.")
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="section-label">dataset</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    ext = file.name.rsplit(".", 1)[-1].lower()
    try:
        return (pd.read_csv(file) if ext == "csv" else pd.read_excel(file)), None
    except Exception as e:
        return None, str(e)

def detect_columns(df):
    num, cat, excl = [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            nu = df[col].nunique()
            (cat if nu <= 10 and nu < len(df) * 0.05 else num).append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            excl.append(col)
        else:
            nu = df[col].nunique()
            (excl if nu == len(df) or (nu > 50 and nu > len(df) * 0.5) else cat).append(col)
    return num, cat, excl

def p_fmt(p):
    return f"{p:.2e}" if p < 0.0001 else f"{p:.5f}"

def plotly_theme():
    return dict(
        template="plotly_dark",
        plot_bgcolor="rgba(13,13,26,0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#c4b5fd"),
        colorway=["#a855f7","#7c3aed","#c084fc","#6d28d9","#ddd6fe"],
    )

def conclusion_card(p_val, alpha, reject_text, fail_text):
    sig = p_val < alpha
    cls = "reject" if sig else "fail-reject"
    badge_cls = "badge-sig" if sig else "badge-insig"
    icon = "✓ REJECT H₀" if sig else "✗ FAIL TO REJECT H₀"
    label = f"{icon}  ·  α = {alpha}"
    text = reject_text if sig else fail_text
    st.markdown(f"""
    <div class="result-card {cls}">
        <span class="badge {badge_cls}">{label}</span>
        <p style="margin-top:8px;font-size:0.97rem;line-height:1.6;color:#e2e8f0;">{text}</p>
    </div>""", unsafe_allow_html=True)


# ── Playground snippets ───────────────────────────────────────────────────────
PLAYGROUND_CODES = {
    "df.head()": "df.head(5)",
    "df.describe()": "df.describe()",
    "df.shape": "df.shape",
    "df.nunique()": "df.nunique()",
    "df.plot.hist()": """\
nc = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in ['index','id','sno','unnamed']]
col = nc[0]
fig, ax = plt.subplots(facecolor='#0d0d1a')
ax.set_facecolor('#0d0d1a')
df[col].dropna().hist(bins=15, alpha=0.8, color='#a855f7', edgecolor='#0d0d1a', ax=ax)
ax.set_title(f"Histogram of {col}", color='#c4b5fd'); ax.set_xlabel(col, color='#94a3b8'); ax.set_ylabel("Frequency", color='#94a3b8')
ax.tick_params(colors='#64748b'); [s.set_color('rgba(139,92,246,0.2)') for s in ax.spines.values()]
plt.tight_layout()""",
    "Central Tendency (Mean, Median, Mode)": """\
nc = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in ['index','id','sno','unnamed']]
col = nc[0]; data = df[col].dropna()
mean, median = data.mean(), data.median()
mode = data.mode().iloc[0] if not data.mode().empty else np.nan
print(f"Column : {col}\\nMean   : {mean:.4f}\\nMedian : {median:.4f}\\nMode   : {mode:.4f}")
fig, ax = plt.subplots(facecolor='#0d0d1a'); ax.set_facecolor('#0d0d1a')
data.hist(bins=20, alpha=0.7, color='#7c3aed', edgecolor='#0d0d1a', ax=ax)
ax.axvline(mean,   color='#f43f5e', ls='--', lw=2, label=f'Mean {mean:.2f}')
ax.axvline(median, color='#10b981',  ls='-',  lw=2, label=f'Median {median:.2f}')
if not np.isnan(mode): ax.axvline(mode, color='#f59e0b', ls=':', lw=2, label=f'Mode {mode:.2f}')
ax.legend(facecolor='#12122a', labelcolor='#c4b5fd')
ax.set_title(f"Central Tendencies — {col}", color='#c4b5fd'); plt.tight_layout()""",
    "Normal Distribution Fit": """\
nc = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in ['index','id','sno','unnamed']]
col = nc[0]; data = df[col].dropna()
mu, sigma = norm.fit(data); sw_stat, sw_p = shapiro(data[:5000])
print(f"Column: {col}\\nμ={mu:.4f}  σ={sigma:.4f}\\nShapiro-Wilk W={sw_stat:.4f}  p={sw_p:.6f}")
print("→ NORMAL (p>0.05)" if sw_p > 0.05 else "→ NOT normal (p≤0.05)")
fig, ax = plt.subplots(facecolor='#0d0d1a'); ax.set_facecolor('#0d0d1a')
data.hist(bins='auto', density=True, alpha=0.6, color='#7c3aed', edgecolor='#0d0d1a', ax=ax)
xs = np.linspace(*ax.get_xlim(), 200)
ax.plot(xs, norm.pdf(xs, mu, sigma), color='#f43f5e', lw=2.5, label=f'Normal μ={mu:.2f} σ={sigma:.2f}')
ax.set_title(f"Normal Fit — {col}", color='#c4b5fd'); ax.legend(facecolor='#12122a', labelcolor='#c4b5fd'); plt.tight_layout()""",
    "Percentiles & Boxplot": """\
nc = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in ['index','id','sno','unnamed']]
col = nc[0]; data = df[col].dropna()
q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
print(f"Column: {col}\\nMin={data.min():.4f}  Q1={q1:.4f}  Median={data.median():.4f}  Q3={q3:.4f}  Max={data.max():.4f}  IQR={q3-q1:.4f}")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True, facecolor='#0d0d1a')
for ax in [ax1, ax2]: ax.set_facecolor('#0d0d1a')
ax1.boxplot(data, vert=False, patch_artist=True,
    boxprops=dict(facecolor='#7c3aed', alpha=0.5), medianprops=dict(color='#f43f5e', lw=2),
    whiskerprops=dict(color='#a78bfa'), capprops=dict(color='#a78bfa'))
ax2.violinplot(data, vert=False, showmeans=True, showmedians=True)
ax2.set_xlabel(col, color='#94a3b8'); fig.suptitle(f"Dispersion — {col}", color='#c4b5fd')
plt.tight_layout()""",
    "Z-Score Outlier Detection": """\
nc = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in ['index','id','sno','unnamed']]
col = nc[0]; data = df[col].dropna(); mean, std = data.mean(), data.std()
z = (data - mean) / std; outliers = data[np.abs(z) > 3]
print(f"Column: {col}\\nMean={mean:.4f}  Std={std:.4f}\\nTotal rows={len(data)}  Outliers(|z|>3)={len(outliers)}")
fig, ax = plt.subplots(facecolor='#0d0d1a'); ax.set_facecolor('#0d0d1a')
data.hist(bins=25, alpha=0.7, color='#7c3aed', edgecolor='#0d0d1a', ax=ax)
ax.axvline(mean, color='#a78bfa', lw=2, label=f'Mean {mean:.2f}')
ax.axvline(mean+3*std, color='#f43f5e', lw=1.5, ls='--', label='+3σ')
ax.axvline(mean-3*std, color='#f43f5e', lw=1.5, ls='--', label='-3σ')
ax.set_title(f"Z-Score Outlier Detection — {col}", color='#c4b5fd')
ax.legend(facecolor='#12122a', labelcolor='#c4b5fd'); plt.tight_layout()""",
    "Linear Regression & Correlation": """\
nc = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in ['index','id','sno','unnamed']]
x_col, y_col = nc[0], nc[1]; data = df[[x_col, y_col]].dropna()
x, y = data[x_col].values, data[y_col].values
slope, intercept, r, p_val, se = linregress(x, y); r2 = r**2
strength = "strong" if abs(r)>0.7 else ("moderate" if abs(r)>0.4 else "weak")
print(f"X={x_col}  Y={y_col}\\nr={r:.4f} ({strength})  R²={r2:.4f}  Y={slope:.4f}X+{intercept:.4f}  p={p_val:.4e}")
print("→ Significant (p<0.05)" if p_val<0.05 else "→ NOT significant")
x_line = np.linspace(x.min(), x.max(), 200)
fig, ax = plt.subplots(facecolor='#0d0d1a'); ax.set_facecolor('#0d0d1a')
ax.scatter(x, y, alpha=0.5, color='#7c3aed', s=20, label='Data', edgecolors='none')
ax.plot(x_line, slope*x_line+intercept, color='#f43f5e', lw=2.5, label=f'Fit r={r:.2f} R²={r2:.2f}')
ax.set_xlabel(x_col, color='#94a3b8'); ax.set_ylabel(y_col, color='#94a3b8')
ax.set_title(f"Linear Regression — {y_col} vs {x_col}", color='#c4b5fd')
ax.legend(facecolor='#12122a', labelcolor='#c4b5fd'); plt.tight_layout()""",
    "One-Sample T-Test": """\
nc = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in ['index','id','sno','unnamed']]
col = nc[0]; data = df[col].dropna(); mu0 = data.mean() * 0.95
t_stat, p = ttest_1samp(data, mu0)
print(f"Column={col}  n={len(data)}  mean={data.mean():.4f}  μ₀={mu0:.4f}  t={t_stat:.4f}  p={p:.6e}")
print("✅ REJECT H₀ (p<0.05)" if p<0.05 else "❌ FAIL TO REJECT H₀")
fig, ax = plt.subplots(facecolor='#0d0d1a'); ax.set_facecolor('#0d0d1a')
data.hist(bins='auto', density=True, alpha=0.6, color='#7c3aed', edgecolor='#0d0d1a', ax=ax)
ax.axvline(data.mean(), color='#10b981', lw=2.5, ls='-',  label=f'Sample mean {data.mean():.2f}')
ax.axvline(mu0,         color='#f43f5e',   lw=2.5, ls='--', label=f'H₀ mean {mu0:.2f}')
ax.set_title(f"One-Sample T-Test — {col}", color='#c4b5fd')
ax.legend(facecolor='#12122a', labelcolor='#c4b5fd'); plt.tight_layout()""",
    "Two-Sample T-Test": """\
nc = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in ['index','id','sno','unnamed']]
cc = list(df.select_dtypes(exclude=[np.number]).columns)
if cc:
    num_col, cat_col = nc[0], cc[0]
    grps = df[cat_col].dropna().unique()[:2]
    g1 = df[df[cat_col]==grps[0]][num_col].dropna(); g2 = df[df[cat_col]==grps[1]][num_col].dropna()
    l1, l2 = str(grps[0]), str(grps[1])
else:
    g1, g2, l1, l2 = df[nc[0]].dropna(), df[nc[1]].dropna(), nc[0], nc[1]
t_stat, p = ttest_ind(g1, g2, equal_var=False)
print(f"{l1}: n={len(g1)} mean={g1.mean():.4f}  {l2}: n={len(g2)} mean={g2.mean():.4f}")
print(f"Welch t={t_stat:.4f}  p={p:.6e}")
print("✅ REJECT H₀" if p<0.05 else "❌ FAIL TO REJECT H₀")
fig, ax = plt.subplots(facecolor='#0d0d1a'); ax.set_facecolor('#0d0d1a')
bp = ax.boxplot([g1, g2], labels=[l1, l2], patch_artist=True,
    boxprops=dict(facecolor='#7c3aed', alpha=0.5), medianprops=dict(color='#f43f5e', lw=2),
    whiskerprops=dict(color='#a78bfa'), capprops=dict(color='#a78bfa'))
ax.set_title(f"Two-Sample T-Test (t={t_stat:.2f}, p={p:.4f})", color='#c4b5fd'); plt.tight_layout()""",
    "One-Way ANOVA": """\
nc = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in ['index','id','sno','unnamed']]
cc = list(df.select_dtypes(exclude=[np.number]).columns)
if not cc: print("Need a categorical column for grouping.")
else:
    num_col, cat_col = nc[0], cc[0]
    grps = df[cat_col].dropna().unique()
    gdata = [df[df[cat_col]==g][num_col].dropna() for g in grps if len(df[df[cat_col]==g][num_col].dropna())>1]
    f_stat, p = f_oneway(*gdata)
    print(f"Factor={cat_col}  Metric={num_col}  F={f_stat:.4f}  p={p:.6e}")
    print("✅ REJECT H₀" if p<0.05 else "❌ FAIL TO REJECT H₀")
    fig, ax = plt.subplots(facecolor='#0d0d1a'); ax.set_facecolor('#0d0d1a')
    ax.boxplot(gdata, labels=[str(g)[:12] for g in grps if len(df[df[cat_col]==g][num_col].dropna())>1],
        patch_artist=True, boxprops=dict(facecolor='#7c3aed', alpha=0.5), medianprops=dict(color='#f43f5e', lw=2),
        whiskerprops=dict(color='#a78bfa'), capprops=dict(color='#a78bfa'))
    ax.set_title(f"One-Way ANOVA — {num_col} by {cat_col}  (F={f_stat:.2f}, p={p:.4f})", color='#c4b5fd')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout()""",
    "Chi-Square Independence Test": """\
cc = list(df.select_dtypes(exclude=[np.number]).columns)
if len(cc)<2: print("Need 2 categorical columns.")
else:
    col1, col2 = cc[0], cc[1]; ct = pd.crosstab(df[col1], df[col2])
    stat, p, dof, exp = chi2_contingency(ct)
    print(f"χ²={stat:.4f}  df={dof}  p={p:.6e}")
    print("✅ REJECT H₀ — DEPENDENT" if p<0.05 else "❌ FAIL TO REJECT H₀ — INDEPENDENT")
    fig, ax = plt.subplots(facecolor='#0d0d1a'); ax.set_facecolor('#0d0d1a')
    im = ax.imshow(ct.values, cmap='Purples')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns, rotation=30, ha='right', color='#94a3b8')
    ax.set_yticks(range(len(ct.index)));   ax.set_yticklabels(ct.index, color='#94a3b8')
    for i in range(len(ct.index)):
        for j in range(len(ct.columns)): ax.text(j, i, ct.values[i,j], ha='center', va='center', fontsize=8, color='white')
    ax.set_title(f"Chi-Square — {col1} vs {col2}  (p={p:.4f})", color='#c4b5fd'); plt.tight_layout()""",
}

PLAYGROUND_HINTS = {
    "df.plot.hist()": "Bell shape → normality; long tails → skewness.",
    "Central Tendency (Mean, Median, Mode)": "Mean > Median → right-skewed; Mean < Median → left-skewed.",
    "Normal Distribution Fit": "Shapiro-Wilk p > 0.05 → data is consistent with normality.",
    "Percentiles & Boxplot": "Box = Q1–Q3 (IQR). Points beyond 1.5×IQR whiskers are potential outliers.",
    "Z-Score Outlier Detection": "~0.27% of normally distributed data falls beyond ±3σ.",
    "Linear Regression & Correlation": "r ∈ [−1, +1] measures linear association. R² = % variance in Y explained by X.",
    "One-Sample T-Test": "Tests whether sample mean equals a known μ₀. p < 0.05 → significant difference.",
    "Two-Sample T-Test": "Welch's t-test (unequal variance). p < 0.05 → groups have different population means.",
    "One-Way ANOVA": "F = between-group / within-group variance. p < 0.05 → at least one group mean differs.",
    "Chi-Square Independence Test": "p < 0.05 → variables are statistically associated (not independent).",
}


# ── Main app logic ─────────────────────────────────────────────────────────────
if uploaded_file is not None:
    df, err = load_data(uploaded_file)
    if err:
        st.error(f"❌ Error loading file: {err}")
    else:
        st.sidebar.success(f"✓ `{uploaded_file.name}`")
        detected_num, detected_cat, _ = detect_columns(df)

        with st.sidebar.expander("⚙️ Column Types"):
            final_num, final_cat = [], []
            for col in df.columns:
                default = "Numerical" if col in detected_num else "Categorical" if col in detected_cat else "Exclude"
                opts = ["Numerical", "Categorical", "Exclude"]
                choice = st.selectbox(f"`{col}`", opts, index=opts.index(default), key=f"ov_{col}")
                if choice == "Numerical": final_num.append(col)
                elif choice == "Categorical": final_cat.append(col)

        # ── Tabs ──────────────────────────────────────────────────────────────
        tabs = st.tabs([
            "📋  Dataset",
            "📈  Correlation",
            "📉  Regression",
            "📊  ANOVA",
            "🔲  Chi-Square",
            "🧪  Playground",
        ])

        # ═════════════════════════════════════════════════════════════════════
        # TAB 1 — DATASET
        # ═════════════════════════════════════════════════════════════════════
        with tabs[0]:
            st.markdown('<div class="section-label">// dataset overview</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            for col_obj, label, val, color in [
                (c1, "Rows",        df.shape[0],    "#e2e8f0"),
                (c2, "Columns",     df.shape[1],    "#e2e8f0"),
                (c3, "Numerical",   len(final_num), "#a855f7"),
                (c4, "Categorical", len(final_cat), "#10b981"),
            ]:
                with col_obj:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-val" style="color:{color};">{val}</div>
                    </div>""", unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="section-label">// raw data preview</div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)

            left, right = st.columns(2)
            with left:
                st.markdown('<div class="section-label">// column info & missing values</div>', unsafe_allow_html=True)
                info = []
                for col in df.columns:
                    miss = df[col].isnull().sum()
                    atype = "Numerical" if col in final_num else "Categorical" if col in final_cat else "Excluded"
                    info.append([col, str(df[col].dtype), atype, f"{miss} ({miss/len(df)*100:.1f}%)"])
                st.table(pd.DataFrame(info, columns=["Column", "Dtype", "Type", "Missing"]))
            with right:
                st.markdown('<div class="section-label">// numerical summary statistics</div>', unsafe_allow_html=True)
                if final_num:
                    st.dataframe(df[final_num].describe().T, use_container_width=True)
                else:
                    st.info("No numerical columns detected.")

        # ═════════════════════════════════════════════════════════════════════
        # TAB 2 — CORRELATION
        # ═════════════════════════════════════════════════════════════════════
        with tabs[1]:
            st.markdown('<div class="section-label">// pearson correlation analysis</div>', unsafe_allow_html=True)
            if len(final_num) < 2:
                st.warning("Need at least 2 numerical columns.")
            else:
                cx1, cx2 = st.columns(2)
                var_x = cx1.selectbox("Variable X", final_num, key="corr_x")
                var_y = cx2.selectbox("Variable Y", [c for c in final_num if c != var_x], key="corr_y")

                cdf = df[[var_x, var_y]].dropna()
                if len(cdf) < 3:
                    st.error("Need at least 3 non-null rows.")
                else:
                    st.markdown('<div class="section-label">// hypothesis</div>', unsafe_allow_html=True)
                    st.latex(r"H_0: \rho = 0 \qquad H_1: \rho \neq 0")

                    r_coef, p_val = stats.pearsonr(cdf[var_x], cdf[var_y])
                    r_sq = r_coef ** 2
                    t_stat = r_coef * np.sqrt((len(cdf) - 2) / (1 - r_sq + 1e-15))

                    rc, pc = st.columns(2)
                    with rc:
                        st.markdown('<div class="section-label">// calculations</div>', unsafe_allow_html=True)
                        st.markdown(f"""<table class="styled-table">
                        <thead><tr><th>Metric</th><th>Value</th><th>Excel</th></tr></thead>
                        <tbody>
                        <tr><td><b>n</b></td><td>{len(cdf)}</td><td><code>COUNT()</code></td></tr>
                        <tr><td><b>r</b></td><td>{r_coef:.5f}</td><td><code>CORREL()</code></td></tr>
                        <tr><td><b>R²</b></td><td>{r_sq:.5f}</td><td><code>RSQ()</code></td></tr>
                        <tr><td><b>t-stat</b></td><td>{t_stat:.5f}</td><td>—</td></tr>
                        <tr><td><b>p-value</b></td><td>{p_fmt(p_val)}</td><td><code>T.DIST.2T()</code></td></tr>
                        </tbody></table>""", unsafe_allow_html=True)
                        strength = "strong" if abs(r_coef)>=0.7 else "moderate" if abs(r_coef)>=0.4 else "weak"
                        direction = "positive" if r_coef > 0 else "negative"
                        st.markdown(f"<br><span style='color:#a78bfa;font-family:JetBrains Mono,monospace;font-size:0.85rem;'>→ {strength} {direction} correlation &nbsp;(r = {r_coef:.3f})</span>", unsafe_allow_html=True)
                    with pc:
                        st.markdown('<div class="section-label">// scatter plot</div>', unsafe_allow_html=True)
                        fig = px.scatter(cdf, x=var_x, y=var_y, trendline="ols",
                            trendline_color_override="#f43f5e")
                        fig.update_layout(**plotly_theme())
                        fig.update_traces(marker=dict(color="#a855f7", size=6, opacity=0.6))
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown('<div class="section-label">// conclusion</div>', unsafe_allow_html=True)
                    conclusion_card(p_val, alpha,
                        f"p-value <b>{p_fmt(p_val)}</b> &lt; α={alpha} → <strong>reject H₀</strong>. Significant linear correlation between <code>{var_x}</code> and <code>{var_y}</code>.",
                        f"p-value <b>{p_fmt(p_val)}</b> ≥ α={alpha} → <strong>fail to reject H₀</strong>. Insufficient evidence of linear correlation between <code>{var_x}</code> and <code>{var_y}</code>.")

        # ═════════════════════════════════════════════════════════════════════
        # TAB 3 — LINEAR REGRESSION
        # ═════════════════════════════════════════════════════════════════════
        with tabs[2]:
            st.markdown('<div class="section-label">// OLS linear regression</div>', unsafe_allow_html=True)
            if len(final_num) < 2:
                st.warning("Need at least 2 numerical columns.")
            else:
                rc1, rc2 = st.columns([1, 2])
                dep_var = rc1.selectbox("Dependent Variable (Y)", final_num, key="reg_y")
                indep_opts = [c for c in final_num if c != dep_var]
                indep_vars = rc1.multiselect("Independent Variables (X)", indep_opts,
                    default=[indep_opts[0]] if indep_opts else [])
                rc2.info("Both Y and X must be **numerical** columns for OLS regression.")

                if indep_vars:
                    clean = df[[dep_var] + indep_vars].dropna()
                    if len(clean) < len(indep_vars) + 2:
                        st.error("Insufficient observations after dropping nulls.")
                    else:
                        X = sm.add_constant(clean[indep_vars].astype(float))
                        y = clean[dep_var]
                        try:
                            model = sm.OLS(y, X).fit()
                        except Exception as e:
                            st.error(f"Regression error: {e}")
                            st.stop()

                        st.markdown('<div class="section-label">// hypothesis</div>', unsafe_allow_html=True)
                        st.latex(r"H_0: \beta_1 = \dots = \beta_p = 0 \qquad H_1: \text{At least one } \beta_j \neq 0")

                        r2, adj_r2 = model.rsquared, model.rsquared_adj
                        f_stat, f_pval = model.fvalue, model.f_pvalue
                        n_obs = int(model.nobs)
                        df_m, df_r = int(model.df_model), int(model.df_resid)
                        std_err = np.sqrt(model.scale)

                        s1, s2 = st.columns(2)
                        with s1:
                            st.markdown('<div class="section-label">// regression statistics</div>', unsafe_allow_html=True)
                            st.markdown(f"""<table class="styled-table">
                            <tr><td><b>Multiple R</b></td><td>{np.sqrt(r2):.5f}</td></tr>
                            <tr><td><b>R²</b></td><td>{r2:.5f}</td></tr>
                            <tr><td><b>Adjusted R²</b></td><td>{adj_r2:.5f}</td></tr>
                            <tr><td><b>Std Error</b></td><td>{std_err:.5f}</td></tr>
                            <tr><td><b>Observations</b></td><td>{n_obs}</td></tr>
                            </table>""", unsafe_allow_html=True)

                        with s2:
                            ss_reg = model.ess; ss_res = model.ssr; ss_tot = ss_reg + ss_res
                            ms_reg = ss_reg/df_m if df_m>0 else 0
                            ms_res = ss_res/df_r if df_r>0 else 0
                            st.markdown('<div class="section-label">// ANOVA table</div>', unsafe_allow_html=True)
                            st.markdown(f"""<table class="styled-table">
                            <thead><tr><th>Source</th><th>df</th><th>SS</th><th>MS</th><th>F</th><th>p-value</th></tr></thead>
                            <tbody>
                            <tr><td><b>Regression</b></td><td>{df_m}</td><td>{ss_reg:.3f}</td><td>{ms_reg:.3f}</td><td>{f_stat:.4f}</td><td>{p_fmt(f_pval)}</td></tr>
                            <tr><td><b>Residual</b></td><td>{df_r}</td><td>{ss_res:.3f}</td><td>{ms_res:.3f}</td><td></td><td></td></tr>
                            <tr><td><b>Total</b></td><td>{df_m+df_r}</td><td>{ss_tot:.3f}</td><td></td><td></td><td></td></tr>
                            </tbody></table>""", unsafe_allow_html=True)

                        st.markdown('<div class="section-label">// coefficients table</div>', unsafe_allow_html=True)
                        coef_rows = []
                        for v in X.columns:
                            ci = model.conf_int().loc[v]
                            coef_rows.append([
                                "Intercept" if v=="const" else v,
                                model.params[v], model.bse[v], model.tvalues[v],
                                model.pvalues[v], ci[0], ci[1]
                            ])
                        cdf2 = pd.DataFrame(coef_rows, columns=["Variable","Coef","Std Err","t-Stat","P-value","Lower 95%","Upper 95%"])
                        st.dataframe(cdf2.style.format({
                            "Coef":"{:.5f}","Std Err":"{:.5f}","t-Stat":"{:.5f}",
                            "P-value": lambda x: f"{x:.2e}" if x<0.0001 else f"{x:.5f}",
                            "Lower 95%":"{:.5f}","Upper 95%":"{:.5f}"
                        }), use_container_width=True)

                        st.markdown('<div class="section-label">// regression equation</div>', unsafe_allow_html=True)
                        terms = []
                        for v in X.columns:
                            c = model.params[v]
                            if v == "const":
                                terms.append(f"{c:.4f}")
                            else:
                                sign = "+" if c >= 0 else "-"
                                terms.append(f"{sign} {abs(c):.4f} \\times \\text{{{v.replace('_',' ')}}}")
                        st.latex(rf"\hat{{Y}} = " + " ".join(terms))

                        st.markdown('<div class="section-label">// diagnostic plots</div>', unsafe_allow_html=True)
                        clean["Predicted"] = model.fittedvalues
                        clean["Residuals"] = model.resid
                        d1, d2 = st.columns(2)
                        with d1:
                            if len(indep_vars) == 1:
                                sorted_c = clean.sort_values(by=indep_vars[0])
                                fig_lf = go.Figure()
                                fig_lf.add_trace(go.Scatter(x=clean[indep_vars[0]], y=clean[dep_var],
                                    mode="markers", name=dep_var, marker=dict(color="#7c3aed", size=7, symbol="diamond", opacity=0.7)))
                                fig_lf.add_trace(go.Scatter(x=sorted_c[indep_vars[0]], y=sorted_c["Predicted"],
                                    mode="lines+markers", name="Predicted",
                                    marker=dict(color="#f43f5e", size=5, symbol="square"),
                                    line=dict(color="#f43f5e", width=2)))
                                fig_lf.update_layout(title=f"{indep_vars[0]} Line Fit",
                                    xaxis_title=indep_vars[0], yaxis_title=dep_var, **plotly_theme())
                            else:
                                mn = min(clean[dep_var].min(), clean["Predicted"].min())
                                mx = max(clean[dep_var].max(), clean["Predicted"].max())
                                fig_lf = px.scatter(clean, x="Predicted", y=dep_var,
                                    title="Actual vs. Predicted")
                                fig_lf.update_traces(marker=dict(color="#a855f7", size=6, opacity=0.7))
                                fig_lf.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                    name="Perfect Fit", line=dict(color="#64748b", dash="dash")))
                                fig_lf.update_layout(**plotly_theme())
                            st.plotly_chart(fig_lf, use_container_width=True)
                        with d2:
                            fig_res = px.scatter(clean, x="Predicted", y="Residuals",
                                title="Residuals vs. Fitted")
                            fig_res.update_traces(marker=dict(color="#a855f7", size=6, opacity=0.7))
                            fig_res.add_hline(y=0, line_dash="dash", line_color="#f43f5e")
                            fig_res.update_layout(**plotly_theme())
                            st.plotly_chart(fig_res, use_container_width=True)

                        st.markdown('<div class="section-label"> conclusion</div>', unsafe_allow_html=True)
                        conclusion_card(f_pval, alpha,
                            f"F-stat p-value <b>{p_fmt(f_pval)}</b> &lt; α={alpha} → <strong>reject H₀</strong>. Model is significant. R² = {r2*100:.2f}% of variance in <code>{dep_var}</code> explained.",
                            f"F-stat p-value <b>{p_fmt(f_pval)}</b> ≥ α={alpha} → <strong>fail to reject H₀</strong>. Model is not statistically significant.")

        # ═════════════════════════════════════════════════════════════════════
        # TAB 4 — ANOVA
        # ═════════════════════════════════════════════════════════════════════
        with tabs[3]:
            st.markdown('<div class="section-label">// one-way analysis of variance</div>', unsafe_allow_html=True)
            if not final_cat or not final_num:
                st.warning("Need at least 1 categorical (grouping) and 1 numerical (response) column.")
            else:
                ac1, ac2 = st.columns(2)
                group_col    = ac1.selectbox("Categorical Grouping Variable (X)", final_cat, key="anova_x")
                response_col = ac2.selectbox("Numerical Response Variable (Y)",   final_num, key="anova_y")

                adf = df[[group_col, response_col]].dropna()
                unique_grps = adf[group_col].unique()
                if len(unique_grps) < 2:
                    st.error("Need at least 2 groups.")
                else:
                    st.markdown('<div class="section-label">// hypothesis</div>', unsafe_allow_html=True)
                    st.latex(r"H_0: \mu_1 = \mu_2 = \dots = \mu_k \qquad H_1: \text{At least one group mean differs}")

                    groups_data = [adf[adf[group_col]==g][response_col].values for g in unique_grps]
                    f_val, p_val = stats.f_oneway(*groups_data)

                    st.markdown('<div class="section-label">// group summary</div>', unsafe_allow_html=True)
                    summary = [[g, len(gd), np.mean(gd), np.std(gd,ddof=1), np.std(gd,ddof=1)/np.sqrt(len(gd))]
                               for g, gd in zip(unique_grps, groups_data)]
                    st.dataframe(pd.DataFrame(summary, columns=["Group","n","Mean","Std Dev","Std Error"])
                                 .style.format({"Mean":"{:.5f}","Std Dev":"{:.5f}","Std Error":"{:.5f}"}),
                                 use_container_width=True)

                    all_vals = adf[response_col].values
                    grand_mean = np.mean(all_vals)
                    ss_b = sum(len(gd) * (np.mean(gd) - grand_mean)**2 for gd in groups_data)
                    ss_t = np.sum((all_vals - grand_mean)**2)
                    ss_w = ss_t - ss_b
                    df_b, df_w = len(unique_grps)-1, len(all_vals)-len(unique_grps)
                    ms_b = ss_b/df_b; ms_w = ss_w/df_w
                    f_crit = stats.f.ppf(1-alpha, df_b, df_w)

                    tc, pc = st.columns([6, 5])
                    with tc:
                        st.markdown('<div class="section-label">// ANOVA table</div>', unsafe_allow_html=True)
                        st.markdown(f"""<table class="styled-table">
                        <thead><tr><th>Source</th><th>SS</th><th>df</th><th>MS</th><th>F</th><th>p-value</th><th>F crit (α={alpha})</th></tr></thead>
                        <tbody>
                        <tr><td><b>Between</b></td><td>{ss_b:.5f}</td><td>{df_b}</td><td>{ms_b:.5f}</td><td>{f_val:.5f}</td><td>{p_fmt(p_val)}</td><td>{f_crit:.5f}</td></tr>
                        <tr><td><b>Within</b></td><td>{ss_w:.5f}</td><td>{df_w}</td><td>{ms_w:.5f}</td><td></td><td></td><td></td></tr>
                        <tr><td><b>Total</b></td><td>{ss_t:.5f}</td><td>{df_b+df_w}</td><td></td><td></td><td></td><td></td></tr>
                        </tbody></table>""", unsafe_allow_html=True)
                    with pc:
                        fig = px.box(adf, x=group_col, y=response_col, points="all", color=group_col,
                            title=f"{response_col} by {group_col}")
                        fig.update_layout(**plotly_theme(), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown('<div class="section-label"> conclusion</div>', unsafe_allow_html=True)
                    conclusion_card(p_val, alpha,
                        f"p-value <b>{p_fmt(p_val)}</b> &lt; α={alpha} → <strong>reject H₀</strong>. At least one group mean of <code>{response_col}</code> differs significantly across <code>{group_col}</code>.",
                        f"p-value <b>{p_fmt(p_val)}</b> ≥ α={alpha} → <strong>fail to reject H₀</strong>. No significant difference in <code>{response_col}</code> across <code>{group_col}</code> groups.")

        # ═════════════════════════════════════════════════════════════════════
        # TAB 5 — CHI-SQUARE
        # ═════════════════════════════════════════════════════════════════════
        with tabs[4]:
            st.markdown('<div class="section-label">// chi-square test of independence</div>', unsafe_allow_html=True)
            if len(final_cat) < 2:
                st.warning("Need at least 2 categorical columns.")
            else:
                cc1, cc2 = st.columns(2)
                chi_x = cc1.selectbox("Categorical Variable A", final_cat, key="chi_x")
                chi_y = cc2.selectbox("Categorical Variable B", [c for c in final_cat if c != chi_x], key="chi_y")

                chi_df = df[[chi_x, chi_y]].dropna()
                obs_table = pd.crosstab(chi_df[chi_x], chi_df[chi_y])
                if obs_table.size < 4:
                    st.error("Each variable needs at least 2 unique levels.")
                else:
                    st.markdown('<div class="section-label">// hypothesis</div>', unsafe_allow_html=True)
                    st.latex(rf"H_0: \text{{{chi_x}}} \text{{ and }} \text{{{chi_y}}} \text{{ are independent}} \qquad H_1: \text{{they are dependent}}")

                    chi2_stat, p_val, dof, expected = stats.chi2_contingency(obs_table)
                    exp_table = pd.DataFrame(expected, index=obs_table.index, columns=obs_table.columns)

                    t1, t2 = st.columns(2)
                    t1.markdown('<div class="section-label">// observed frequencies</div>', unsafe_allow_html=True)
                    t1.dataframe(obs_table, use_container_width=True)
                    t2.markdown('<div class="section-label">// expected frequencies</div>', unsafe_allow_html=True)
                    t2.dataframe(exp_table.style.format("{:.2f}"), use_container_width=True)

                    rc, pc = st.columns(2)
                    with rc:
                        st.markdown('<div class="section-label">// test statistics</div>', unsafe_allow_html=True)
                        cells_lt5 = (expected < 5).sum()
                        st.markdown(f"""<table class="styled-table">
                        <thead><tr><th>Statistic</th><th>Value</th><th>Excel</th></tr></thead>
                        <tbody>
                        <tr><td><b>χ²</b></td><td>{chi2_stat:.5f}</td><td><code>CHISQ.TEST()</code></td></tr>
                        <tr><td><b>df</b></td><td>{dof}</td><td><code>(R-1)×(C-1)</code></td></tr>
                        <tr><td><b>p-value</b></td><td>{p_fmt(p_val)}</td><td><code>CHISQ.DIST.RT()</code></td></tr>
                        </tbody></table>""", unsafe_allow_html=True)
                        if cells_lt5 > 0:
                            pct = cells_lt5 / expected.size * 100
                            (st.warning if pct > 20 else st.info)(
                                f"{cells_lt5}/{expected.size} expected cells ({pct:.1f}%) < 5 — Cochran's rule: keep below 20%.")
                    with pc:
                        obs_melt = obs_table.reset_index().melt(id_vars=chi_x, value_name="Observed")
                        exp_melt = exp_table.reset_index().melt(id_vars=chi_x, value_name="Expected")
                        plot_df = pd.merge(obs_melt, exp_melt, on=[chi_x, chi_y]).melt(
                            id_vars=[chi_x, chi_y], value_vars=["Observed","Expected"],
                            var_name="Type", value_name="Count")
                        fig = px.bar(plot_df, x=chi_x, y="Count", color="Type", barmode="group",
                            facet_col=chi_y, title="Observed vs Expected",
                            color_discrete_map={"Observed":"#a855f7","Expected":"#475569"})
                        fig.update_layout(**plotly_theme())
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown('<div class="section-label"> conclusion</div>', unsafe_allow_html=True)
                    conclusion_card(p_val, alpha,
                        f"p-value <b>{p_fmt(p_val)}</b> &lt; α={alpha} → <strong>reject H₀</strong>. Significant association between <code>{chi_x}</code> and <code>{chi_y}</code>.",
                        f"p-value <b>{p_fmt(p_val)}</b> ≥ α={alpha} → <strong>fail to reject H₀</strong>. No significant association between <code>{chi_x}</code> and <code>{chi_y}</code>.")

        # ═════════════════════════════════════════════════════════════════════
        # TAB 6 — PLAYGROUND
        # ═════════════════════════════════════════════════════════════════════
        with tabs[5]:
            st.markdown('<div class="section-label">// interactive stats playground</div>', unsafe_allow_html=True)
            st.markdown("<span style='color:#64748b;font-size:0.85rem;'>Select a test, tweak the code, re-run live on your dataset.</span>", unsafe_allow_html=True)
            st.write("")

            selected = st.selectbox("Function / Test:", list(PLAYGROUND_CODES.keys()))
            st.markdown("---")
            col_code, col_out = st.columns(2)

            with col_code:
                st.markdown('<div class="section-label">// code editor</div>', unsafe_allow_html=True)
                code_input = st.text_area("Edit & Ctrl+Enter to run:",
                    value=PLAYGROUND_CODES[selected], height=380, key=f"pg_{selected}")

            with col_out:
                st.markdown('<div class="section-label">// live output</div>', unsafe_allow_html=True)
                env = {
                    '__builtins__': builtins,
                    'df': df, 'pd': pd, 'np': np, 'plt': plt,
                    'norm': norm, 'binom': binom, 'poisson': poisson,
                    't': t_dist, 'f': f_dist,
                    'shapiro': shapiro,
                    'ttest_1samp': ttest_1samp, 'ttest_ind': ttest_ind, 'ttest_rel': ttest_rel,
                    'f_oneway': f_oneway,
                    'chisquare': chisquare, 'chi2_contingency': chi2_contingency,
                    'linregress': linregress,
                }
                buf = io.StringIO()
                sys.stdout = buf
                err_msg = None
                result = None
                try:
                    lines = [ln for ln in code_input.split('\n') if ln.strip()]
                    if lines:
                        last = lines[-1]
                        rest = "\n".join(lines[:-1])
                        if rest.strip():
                            exec(rest, env)
                        try:
                            result = eval(last, env)
                        except Exception:
                            exec(last, env)
                    else:
                        exec(code_input, env)
                except Exception as e:
                    err_msg = str(e)
                finally:
                    sys.stdout = sys.__stdout__

                if err_msg:
                    st.error(f"Error: {err_msg}")
                else:
                    figs = plt.get_fignums()
                    if figs:
                        for fn in figs:
                            st.pyplot(plt.figure(fn))
                        plt.close('all')
                    printed = buf.getvalue()
                    if printed.strip():
                        st.code(printed, language="text")
                    if result is not None:
                        st.write(result)
                    if not printed.strip() and not figs and result is None:
                        st.info("Executed with no output.")

            if selected in PLAYGROUND_HINTS:
                st.markdown("---")
                st.markdown(f"""
                <div style="background:rgba(109,40,217,0.08);border:1px solid rgba(139,92,246,0.25);border-radius:10px;padding:0.9rem 1.2rem;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#7c3aed;letter-spacing:0.1em;">// INFERENCE</span><br>
                    <span style="color:#c4b5fd;font-size:0.88rem;">{PLAYGROUND_HINTS[selected]}</span>
                </div>""", unsafe_allow_html=True)

# ── Landing page (no file uploaded) ───────────────────────────────────────────
else:
    st.markdown("""
    <div class="upload-box">
        <div style="font-size:3rem;margin-bottom:0.75rem;">📂</div>
        <h3>Upload your dataset to begin</h3>
        <p>Supports <code>.csv</code> and <code>.xlsx</code> — use the sidebar uploader on the left</p>
    </div>

    <div class="landing-grid">
        <div class="landing-card">
            <div class="landing-icon">📈</div>
            <h4>Correlation Analysis</h4>
            <p>Pearson r, R², t-stat, p-value with scatter plot</p>
        </div>
        <div class="landing-card">
            <div class="landing-icon">📉</div>
            <h4>OLS Regression</h4>
            <p>ANOVA table, coefficients, line fit & residuals</p>
        </div>
        <div class="landing-card">
            <div class="landing-icon">📊</div>
            <h4>One-Way ANOVA</h4>
            <p>F-value, SS, MS, F-critical with group boxplots</p>
        </div>
        <div class="landing-card">
            <div class="landing-icon">🔲</div>
            <h4>Chi-Square Test</h4>
            <p>Contingency tables, observed vs expected, χ²</p>
        </div>
        <div class="landing-card">
            <div class="landing-icon">🧪</div>
            <h4>Stats Playground</h4>
            <p>Live code editor — run T-tests, ANOVA, fits, plots</p>
        </div>
        <div class="landing-card">
            <div class="landing-icon">🤖</div>
            <h4>Auto H₀ / H₁</h4>
            <p>Hypotheses generated automatically with conclusions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
