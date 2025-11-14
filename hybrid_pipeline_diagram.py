# hybrid_pipeline_diagram.py
# Draw a clean pipeline diagram (English-only) with matplotlib
# Saves: hybrid_pipeline_en.png / hybrid_pipeline_en.svg

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle
from matplotlib import patheffects as pe

# --- helpers ---------------------------------------------------------------
def add_box(ax, xy, w, h, text, fontsize=13, lw=1.6, fc="white", ec="black", alpha=1.0):
    """Add a rounded box with centered multi-line text."""
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.015,rounding_size=8",
                         linewidth=lw, facecolor=fc, edgecolor=ec, alpha=alpha)
    ax.add_patch(box)
    # Centered text inside the box
    ax.text(x + w/2, y + h/2, text, fontsize=fontsize, ha="center", va="center",
            linespacing=1.25, wrap=True,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    return box

def arrow(ax, src_center, dst_center, lw=1.6):
    """Draw a curved arrow from src_center -> dst_center."""
    style = ArrowStyle("Simple", head_width=8, head_length=10)
    ax.annotate("",
                xy=dst_center, xycoords="data",
                xytext=src_center, textcoords="data",
                arrowprops=dict(arrowstyle=style, lw=lw, color="black",
                                shrinkA=8, shrinkB=8, connectionstyle="arc3,rad=0.0"))

# --- canvas ----------------------------------------------------------------
plt.figure(figsize=(10.5, 12), dpi=150)  # big, readable
ax = plt.gca()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title
ax.text(50, 97, "Hybrid Modeling Pipeline (SINDy + LightGBM)", ha="center", va="center",
        fontsize=18, weight="bold")

# Layout constants
W = 72      # main box width
H = 13      # main box height
Xc = 14     # left margin for centered boxes -> x = (100 - W)/2 ≈ 14
G = 7       # vertical gap

# STEP 1 (top, centered)
b1 = add_box(
    ax, (Xc, 80), W, H,
    "Step 1. Build Dynamic State Variables\n"
    "Construct per-user time-series trajectories of five state variables\n"
    "(e.g., balance exhaustion, transaction frequency) defined in Sec. 3.2.",
    fontsize=14
)

# STEP 2 (two parallel boxes: normal & fraud SINDy)
w2 = 34
h2 = 13
x2_left  = 8
x2_right = 100 - 8 - w2
y2 = 80 - H - G

b2a = add_box(
    ax, (x2_left, y2), w2, h2,
    r"Step 2-A. Train $SINDy_{normal}$"+"\n"
    "Identify a dynamics model that best explains\n"
    "normal-user trajectories.",
    fontsize=14
)
b2b = add_box(
    ax, (x2_right, y2), w2, h2,
    r"Step 2-B. Train $SINDy_{fraud}$"+"\n"
    "Identify a dynamics model that best explains\n"
    "fraud-user trajectories.",
    fontsize=14
)

# STEP 3 (centered)
y3 = y2 - h2 - G
b3 = add_box(
    ax, (Xc, y3), W, H,
    r"Step 3. Engineer SINDy-derived Features"+"\n"
    r"Compute per-transaction errors (MSE) from $SINDy_{normal}$ and "
    r"$SINDy_{fraud}$, plus their ratios/differences, to form a new "
    r"SINDy feature group.",
    fontsize=14
)

# STEP 4 (centered)
y4 = y3 - H - G
b4 = add_box(
    ax, (Xc, y4), W, H,
    "Step 4. Train the Hybrid Classifier\n"
    "Concatenate static features (amount, oldbalanceOrg, …) with "
    "SINDy-derived features, then train a LightGBM classifier.",
    fontsize=14
)

# STEP 5 (centered)
y5 = y4 - H - G
b5 = add_box(
    ax, (Xc, y5), W, H,
    "Step 5. Evaluation & XAI Analysis\n"
    "Compare baseline (static-only) vs. hybrid (static + SINDy) using "
    "AP and ROC-AUC. Analyze decisions with SHAP, PDP/ICE, and LIME.",
    fontsize=14
)

# --- arrows (centers) ------------------------------------------------------
def center_of(box):
    p = box.get_bbox().get_points()
    (x0, y0), (x1, y1) = p
    return ((x0 + x1)/2, (y0 + y1)/2)

c1  = center_of(b1)
c2a = center_of(b2a)
c2b = center_of(b2b)
c3  = center_of(b3)
c4  = center_of(b4)
c5  = center_of(b5)

# 1 -> 2A and 2B
arrow(ax, (c1[0], c1[1] - H/2), (c2a[0], c2a[1] + h2/2))
arrow(ax, (c1[0], c1[1] - H/2), (c2b[0], c2b[1] + h2/2))
# 2A -> 3 and 2B -> 3
arrow(ax, (c2a[0], c2a[1] - h2/2), (c3[0] - W*0.18, c3[1] + H/2))
arrow(ax, (c2b[0], c2b[1] - h2/2), (c3[0] + W*0.18, c3[1] + H/2))
# 3 -> 4 -> 5
arrow(ax, (c3[0], c3[1] - H/2), (c4[0], c4[1] + H/2))
arrow(ax, (c4[0], c4[1] - H/2), (c5[0], c5[1] + H/2))

# Footer notes (small but readable)
ax.text(50, 3.5,
        "Notes: SINDy-derived features include MSE_normal, MSE_fraud, ratios, and first differences.\n"
        "Hybrid = Static features + SINDy features → LightGBM. Evaluation uses AP/ROC-AUC and XAI (SHAP/PDP/LIME).",
        fontsize=10.5, ha="center", va="center")

plt.tight_layout()
plt.savefig("hybrid_pipeline_en.png", bbox_inches="tight")
plt.savefig("hybrid_pipeline_en.svg", bbox_inches="tight")
print("Saved: hybrid_pipeline_en.png / hybrid_pipeline_en.svg")
