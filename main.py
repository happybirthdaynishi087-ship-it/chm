import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heroin · Stereochemistry Lab",
    page_icon="⚗",
    layout="wide",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Playfair+Display:wght@400;700;900&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a27;
    --accent: #c8a96e;
    --accent2: #7eb8c8;
    --r-col: #e07b6a;
    --s-col: #6ab5e0;
    --text: #e8e4dc;
    --muted: #888;
    --border: #2a2a3a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--accent) !important; }

.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
}
.metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    font-family: 'Playfair Display', serif;
}
.metric-sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

.r-badge {
    display: inline-block;
    background: rgba(224,123,106,0.15);
    border: 1px solid var(--r-col);
    color: var(--r-col);
    border-radius: 3px;
    padding: 0.1rem 0.5rem;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    margin: 0.15rem;
}
.s-badge {
    display: inline-block;
    background: rgba(106,181,224,0.15);
    border: 1px solid var(--s-col);
    color: var(--s-col);
    border-radius: 3px;
    padding: 0.1rem 0.5rem;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    margin: 0.15rem;
}

.section-rule {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

.info-box {
    background: var(--surface);
    border-left: 3px solid var(--accent2);
    padding: 1rem 1.2rem;
    border-radius: 0 4px 4px 0;
    font-size: 0.82rem;
    line-height: 1.7;
    color: var(--text);
    margin: 0.8rem 0;
}

.chiral-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
    margin-top: 0.8rem;
}
.chiral-table th {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.65rem;
    border-bottom: 1px solid var(--border);
    padding: 0.5rem 0.8rem;
    text-align: left;
}
.chiral-table td {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid var(--border);
}
.chiral-table tr:hover td { background: var(--surface2); }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--accent);
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
}
.formula-box {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 0.4rem 1rem;
    border-radius: 3px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--accent2);
    margin-top: 0.6rem;
    letter-spacing: 0.05em;
}

div[data-testid="stPlotlyChart"], div[data-testid="stImage"] {
    border-radius: 6px;
    overflow: hidden;
}

/* Streamlit overrides */
.stMarkdown p { color: var(--text) !important; font-family: 'Space Mono', monospace !important; font-size: 0.85rem !important; }
[data-testid="stExpander"] { border: 1px solid var(--border) !important; background: var(--surface) !important; }
button[kind="secondary"] { border-color: var(--border) !important; color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)


# ─── MOLECULE DATA ────────────────────────────────────────────────────────────
# Heroin (diacetylmorphine) - 2D skeletal coordinates, hand-crafted
# Morphine core with two acetyl groups at C3 and C6
# Atom list: (symbol, x, y, label, chiral)
atoms = {
    # Ring atoms - morphine core (approximate 2D positions)
    "C1":  (4.0,  6.0,  "C1",   None),
    "C2":  (3.0,  6.8,  "C2",   None),
    "C3":  (2.0,  6.0,  "C3",   None),   # O-acetyl attached
    "C4":  (2.0,  4.8,  "C4",   None),
    "C5":  (3.0,  4.0,  "C5",   "R"),   # R centre
    "C6":  (4.0,  4.8,  "C6",   "S"),   # S centre
    "C7":  (5.0,  4.0,  "C7",   None),
    "C8":  (5.0,  2.8,  "C8",   None),
    "C9":  (4.0,  2.0,  "C9",   "R"),   # R centre
    "C10": (3.0,  2.8,  "C10",  None),
    "C13": (6.0,  4.0,  "C13",  "S"),  # S centre
    "C14": (6.0,  2.8,  "C14",  None),
    "N":   (6.0,  5.2,  "N",    None),
    # Oxygen atoms
    "O3":  (1.0,  6.8,  "O",    None),   # O at C3 - acetyl
    "O6":  (4.0,  3.4,  "O",    None),   # O at C6 (bridge)
    "O17": (5.0,  5.6,  "O",    None),   # Ether O bridge
    # Acetyl at O3: C3-O-C(=O)-CH3
    "AcC3": (0.0,  6.0,  "C",   None),
    "AcO3": (-0.8, 6.8,  "O",   None),   # =O
    "AcMe3":(0.0,  4.8,  "CH₃", None),
    # Acetyl at C6: C6-O-C(=O)-CH3
    "AcC6": (4.0,  2.4,  "C",   None),  # Moved down
    "AcO6": (3.2,  1.8,  "O",   None),  # =O
    "AcMe6":(4.8,  1.8,  "CH₃", None),
    # N-methyl
    "NMe":  (7.0,  5.6,  "CH₃", None),
}

# Bonds: (a1, a2, order)  order: 1=single, 2=double, 3=aromatic(dashed)
bonds = [
    # Aromatic ring C1-C2-C3-C4
    ("C1","C2", 1.5),
    ("C2","C3", 1.5),
    ("C3","C4", 1.5),
    ("C4","C5", 1),
    ("C5","C6", 1),
    ("C6","C1", 1),
    # Ring bridges
    ("C5","C10",1),
    ("C10","C9",1),
    ("C9","C8", 1),
    ("C8","C7", 1),
    ("C7","C6", 1),
    ("C7","C13",1),
    ("C13","C14",1),
    ("C14","C9", 1),
    ("C13","N",  1),
    ("N","C6",   1),
    # Oxygen bridge
    ("C4","O17", 1),
    ("O17","C14",1),
    # O6 bridge
    ("C6","O6",  1),
    ("O6","AcC6",1),
    # O3 acetyl
    ("C3","O3",  1),
    ("O3","AcC3",1),
    # Acetyl carbonyls
    ("AcC3","AcO3", 2),
    ("AcC3","AcMe3",1),
    ("AcC6","AcO6", 2),
    ("AcC6","AcMe6",1),
    # N-methyl
    ("N","NMe",  1),
]

chiral_centres = {
    "C5":  {"config": "R", "pos": atoms["C5"][:2],  "desc": "C-5",  "note": "Allylic carbon; part of D-ring"},
    "C6":  {"config": "S", "pos": atoms["C6"][:2],  "desc": "C-6",  "note": "Bridgehead; acetyl ether oxygen"},
    "C9":  {"config": "R", "pos": atoms["C9"][:2],  "desc": "C-9",  "note": "Part of C/D-ring junction"},
    "C13": {"config": "S", "pos": atoms["C13"][:2], "desc": "C-13", "note": "N-bearing bridgehead"},
}


def draw_molecule(highlight=None):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    ax.set_aspect("equal")

    # ── Draw bonds ──────────────────────────────────────────────────
    for a1, a2, order in bonds:
        if a1 not in atoms or a2 not in atoms:
            continue
        x1, y1 = atoms[a1][:2]
        x2, y2 = atoms[a2][:2]
        if order == 2:
            # Offset double bond
            dx, dy = x2-x1, y2-y1
            norm = np.sqrt(dx**2+dy**2)
            ox, oy = -dy/norm*0.08, dx/norm*0.08
            ax.plot([x1+ox, x2+ox],[y1+oy, y2+oy], color="#4a4a6a", lw=1.4, zorder=1)
            ax.plot([x1-ox, x2-ox],[y1-oy, y2-oy], color="#4a4a6a", lw=1.4, zorder=1)
        elif order == 1.5:
            # Aromatic — alternating dash
            ax.plot([x1,x2],[y1,y2], color="#3a3a5a", lw=1.6, zorder=1)
            ax.plot([x1,x2],[y1,y2], color="#6a6aaa", lw=0.7,
                    linestyle=(0,(4,3)), zorder=1)
        else:
            ax.plot([x1,x2],[y1,y2], color="#3d3d5c", lw=1.6, zorder=1)

    # ── Draw atoms ──────────────────────────────────────────────────
    for key, data in atoms.items():
        x, y, lbl, chiral = data
        is_chiral = chiral is not None

        if is_chiral:
            config = chiral  # R or S
            ring_col  = "#e07b6a" if config == "R" else "#6ab5e0"
            glow_col  = "#e07b6a44" if config == "R" else "#6ab5e044"
            if highlight and key != highlight:
                ring_col = "#333355"
                glow_col = "#00000000"
            # Glow halo
            glow = Circle((x, y), 0.28, color=glow_col, zorder=2)
            ax.add_patch(glow)
            # Ring
            ring = Circle((x, y), 0.2, fill=False, edgecolor=ring_col,
                          linewidth=1.8, zorder=3)
            ax.add_patch(ring)
            # Config label above atom
            ax.text(x, y+0.38, f"({config})", fontsize=6.5, ha="center",
                    va="bottom", color=ring_col, fontweight="bold",
                    fontfamily="monospace", zorder=5)
        
        # Atom label
        if lbl in ("C", "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C13","C14"):
            pass  # Carbon atoms: show label only if chiral or special
        
        if lbl not in ("C",):
            color = "#e8e4dc"
            if lbl == "N":   color = "#9b9bff"
            elif lbl == "O": color = "#ff9999"
            elif "CH₃" in lbl: color = "#aaaacc"
            
            bg = FancyBboxPatch((x-0.22, y-0.16), 0.44, 0.32,
                                boxstyle="round,pad=0.04",
                                facecolor="#0a0a0f", edgecolor="none", zorder=3)
            ax.add_patch(bg)
            ax.text(x, y, lbl, fontsize=7.5 if "CH₃" in lbl else 9,
                    ha="center", va="center", color=color,
                    fontweight="bold", fontfamily="monospace", zorder=4)

        # Carbon labels for chiral centres
        if is_chiral:
            cnum = key.replace("C","")
            ax.text(x-0.06, y, f"C{cnum}", fontsize=6, ha="center", va="center",
                    color="#aaaacc", fontfamily="monospace", zorder=4)

    # ── Legend ───────────────────────────────────────────────────────
    ax.text(0.02, 0.98, "● R-centre", transform=ax.transAxes,
            fontsize=7.5, va="top", color="#e07b6a", fontfamily="monospace")
    ax.text(0.02, 0.93, "● S-centre", transform=ax.transAxes,
            fontsize=7.5, va="top", color="#6ab5e0", fontfamily="monospace")

    ax.set_xlim(-1.8, 8.0)
    ax.set_ylim(0.8, 8.0)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    return fig


def draw_rs_wheel():
    """CIP priority wheel diagram"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.patch.set_facecolor("#0a0a0f")
    
    configs = [
        ("R", "#e07b6a", "Clockwise\npriority order", axes[0]),
        ("S", "#6ab5e0", "Counterclockwise\npriority order", axes[1]),
    ]
    
    for config, col, desc, ax in configs:
        ax.set_facecolor("#12121a")
        ax.set_aspect("equal")
        
        # Central circle
        c = Circle((0,0), 0.35, color=col, alpha=0.2, zorder=1)
        ax.add_patch(c)
        ax.text(0,0, config, ha="center", va="center", fontsize=22,
                fontweight="bold", color=col, fontfamily="serif", zorder=2)
        
        # Priority arrows
        angles = [90, 210, 330] if config == "R" else [90, 330, 210]
        labels = ["1", "2", "3"]
        
        for i, (ang, lbl) in enumerate(zip(angles, labels)):
            rad = np.radians(ang)
            ex, ey = 0.85*np.cos(rad), 0.85*np.sin(rad)
            ax.text(ex*1.35, ey*1.35, lbl, ha="center", va="center",
                    fontsize=11, color=col, fontfamily="monospace",
                    fontweight="bold")
            ax.annotate("", xy=(ex*0.42, ey*0.42), xytext=(ex*0.9, ey*0.9),
                        arrowprops=dict(arrowstyle="->" if config=="R" else "<-",
                                        color=col, lw=1.5))
        
        ax.text(0, -1.35, desc, ha="center", va="top", fontsize=7.5,
                color="#888", fontfamily="monospace")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.axis("off")
    
    fig.tight_layout(pad=0.5)
    return fig


# ─── LAYOUT ──────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="padding: 1.8rem 0 0.5rem 0; border-bottom: 1px solid #2a2a3a; margin-bottom: 1.5rem;">
  <div class="hero-title">Diacetylmorphine</div>
  <div class="hero-sub">Stereochemistry · Chiral Analysis · Isomer Count</div>
  <div class="formula-box">C₂₁H₂₃NO₅ · MW 369.41 g/mol</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("### Structural Map")
    st.markdown('<p style="color:#888;font-size:0.75rem;margin-top:-0.5rem;">2D skeletal representation · chiral centres highlighted</p>', unsafe_allow_html=True)
    
    highlight_opt = st.selectbox(
        "Highlight centre", 
        ["None", "C5 (R)", "C6 (S)", "C9 (R)", "C13 (S)"],
        index=0,
        label_visibility="collapsed"
    )
    hl_map = {"None": None, "C5 (R)": "C5", "C6 (S)": "C6", "C9 (R)": "C9", "C13 (S)": "C13"}
    fig_mol = draw_molecule(highlight=hl_map[highlight_opt])
    st.pyplot(fig_mol, use_container_width=True)
    plt.close(fig_mol)

with col_right:
    st.markdown("### Stereochemical Data")
    
    # Metrics
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Chiral Centres</div>
        <div class="metric-value">4</div>
        <div class="metric-sub">C-5 · C-6 · C-9 · C-13</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Theoretical Stereoisomers</div>
        <div class="metric-value">2⁴ = 16</div>
        <div class="metric-sub">Enantiomers + Diastereomers</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Natural Configuration</div>
        <div class="metric-value" style="font-size:1.1rem;">5R, 6S, 9R, 13S</div>
        <div class="metric-sub">Only naturally-occurring form · (–)-enantiomer</div>
    </div>
    <div class="metric-card" style="border-left-color: #6ab5e0;">
        <div class="metric-label">Optical Rotation</div>
        <div class="metric-value">–166°</div>
        <div class="metric-sub">[α]²⁰_D in chloroform · levorotatory</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr class="section-rule"/>', unsafe_allow_html=True)
    
    # Chiral centre table
    st.markdown("**Chiral Centres**", unsafe_allow_html=True)
    st.markdown("""
    <table class="chiral-table">
      <thead><tr>
        <th>Centre</th><th>Config</th><th>Note</th>
      </tr></thead>
      <tbody>
        <tr><td>C-5</td>
            <td><span class="r-badge">R</span></td>
            <td style="color:#888">Allylic; D-ring</td></tr>
        <tr><td>C-6</td>
            <td><span class="s-badge">S</span></td>
            <td style="color:#888">Acetyl ether O bridge</td></tr>
        <tr><td>C-9</td>
            <td><span class="r-badge">R</span></td>
            <td style="color:#888">C/D-ring junction</td></tr>
        <tr><td>C-13</td>
            <td><span class="s-badge">S</span></td>
            <td style="color:#888">N-bearing bridgehead</td></tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

# ── CIP Rule diagram ──────────────────────────────────────────────────────────
st.markdown('<hr class="section-rule"/>', unsafe_allow_html=True)
col_a, col_b = st.columns([2, 3], gap="large")

with col_a:
    st.markdown("### R / S Assignment Rule")
    st.markdown("""
    <div class="info-box">
      <b style="color:#c8a96e">CIP Priority Rules</b><br><br>
      1. Rank substituents by atomic number (high → low = 1→4)<br>
      2. Orient molecule so priority <b>4</b> points <i>away</i><br>
      3. Trace 1→2→3:<br>
      &nbsp;&nbsp;· Clockwise = <span style="color:#e07b6a"><b>R</b></span> (Rectus)<br>
      &nbsp;&nbsp;· Counter-clockwise = <span style="color:#6ab5e0"><b>S</b></span> (Sinister)<br><br>
      In heroin's rigid polycyclic framework, the bridged ring system <b>fixes</b> all four centres in the 5R,6S,9R,13S arrangement.
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("### CIP Rotation Diagrams")
    fig_wheel = draw_rs_wheel()
    st.pyplot(fig_wheel, use_container_width=True)
    plt.close(fig_wheel)

# ── Isomer count explainer ────────────────────────────────────────────────────
st.markdown('<hr class="section-rule"/>', unsafe_allow_html=True)
st.markdown("### Stereoisomer Count")

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown("""
    <div class="metric-card" style="border-left-color:#9b9bff">
        <div class="metric-label">Formula</div>
        <div class="metric-value" style="font-size:1.3rem;">2ⁿ</div>
        <div class="metric-sub">n = number of chiral centres<br>n = 4 → 2⁴ = <b>16</b> max</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card" style="border-left-color:#e07b6a">
        <div class="metric-label">Enantiomers</div>
        <div class="metric-value">2</div>
        <div class="metric-sub">(5R,6S,9R,13S) — natural<br>(5S,6R,9S,13R) — mirror image</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card" style="border-left-color:#6ab5e0">
        <div class="metric-label">Diastereomers</div>
        <div class="metric-value">14</div>
        <div class="metric-sub">Remaining 14 of 16<br>differ at ≥1 centre only</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="info-box" style="margin-top:1rem">
  <b style="color:#c8a96e">Why only 1 biologically active form?</b><br>
  Enzymes are chiral catalysts — they fit only the natural (5R,6S,9R,13S) enantiomer into their active sites.
  The (+)-enantiomer (5S,6R,9S,13R) has negligible opioid activity. The rigid tricyclic/tetracyclic 
  scaffold of the morphinan skeleton also means many of the 16 theoretical isomers 
  are <i>geometrically strained or chemically impossible</i> to synthesise.
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding-top:1rem; border-top:1px solid #2a2a3a; 
     font-size:0.65rem; color:#444; letter-spacing:0.1em; text-transform:uppercase;">
  ⚗ Stereochemistry Lab · Diacetylmorphine C₂₁H₂₃NO₅ · 4 Chiral Centres · 16 Theoretical Stereoisomers
</div>
""", unsafe_allow_html=True)
