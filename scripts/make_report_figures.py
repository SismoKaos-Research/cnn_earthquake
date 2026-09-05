"""Figures for the TUBITAK report. Static print output, not interactive.

Three figures, each chosen by what its data has to do:

  Sekil 1  recall by log SNR vs by magnitude -- two small multiples sharing a
           y-axis, because the two binnings are different quantities and a
           dual x-axis would be unreadable. Bars, not lines: the bins are
           ordered categories of unequal width, and a line would imply
           continuity between them.
  Sekil 2  raw AUC vs headroom captured -- the report's central methodological
           claim. Both panels use a FULL axis: the point is that raw AUC looks
           flat near 1.0 while headroom spreads, and truncating the first
           panel would manufacture the difference the figure exists to say is
           invisible.
  Sekil 3  fusion minus 2B across four datasets -- a polarity job, so a
           diverging treatment with a zero line and two poles.

Palette is the validated default (blue #2a78d6, orange #eb6834); the aqua slot
carries a contrast warning against a light surface, so every bar is directly
labelled rather than relying on fill alone.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a86"
INK, MUTED = "#0b0b0b", "#52514e"
OUT = "/home/hogib/Desktop/rapor_sekiller"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#c9c9c5", "axes.linewidth": 0.8,
    "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def tidy(ax, ygrid=True):
    """Recessive axes: no top/right spine, grid behind the marks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color="#e8e8e5", linewidth=0.7)
        ax.set_axisbelow(True)


def fig1():
    """Recall by SNR and by magnitude, shared y."""
    snr_lab = ["< −2,0", "−2,0 – 0,72", "0,72 – 3,42", "> 3,42"]
    snr_val = [0.8857, 0.8696, 0.9859, 1.0000]
    snr_n = [35, 2630, 4261, 976]
    mag_lab = ["1,5–2,0", "2,0–2,5", "2,5–3,0", "3,0–3,5", "> 3,5"]
    mag_val = [0.9100, 0.9458, 0.9792, 0.9900, 0.9850]
    mag_n = [1611, 4152, 1445, 498, 200]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, lab, val, ns, colour, title in (
            (axes[0], snr_lab, snr_val, snr_n, BLUE, "log SNR'ye göre"),
            (axes[1], mag_lab, mag_val, mag_n, ORANGE, "Büyüklüğe göre")):
        ax.bar(range(len(val)), val, color=colour, width=0.62, zorder=3)
        for i, (v, n) in enumerate(zip(val, ns)):
            ax.text(i, v + 0.004, f"{v:.4f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=8, color=INK)
            ax.text(i, 0.858, f"n={n:,}".replace(",", "."), ha="center",
                    va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(range(len(lab)))
        ax.set_xticklabels(lab, fontsize=8)
        ax.set_title(title, fontsize=9, color=INK, pad=8)
        ax.axhline(0.9484, color=GREY, linestyle="--", linewidth=0.9, zorder=2)
        tidy(ax)
    axes[0].set_ylim(0.85, 1.012)
    axes[0].set_ylabel("Recall")
    axes[0].text(-0.42, 0.9484, "genel 0,9484", fontsize=7, color=MUTED,
                 va="bottom", ha="left")
    fig.savefig(f"{OUT}/sekil1_isletim_zarfi.png")
    plt.close(fig)


def fig2():
    """Raw AUC vs headroom captured -- both on full axes."""
    lab = ["EQTransformer\n(6 s pencere)", "EQTransformer\n(tam 60 s)",
           "Bu proje\n(katalog sabitli)"]
    auc = [0.9989, 0.9976, 0.9971]
    head = [95.6, 90.3, 88.3]
    cols = [ORANGE, ORANGE, BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    ax = axes[0]
    ax.bar(range(3), auc, color=cols, width=0.55, zorder=3)
    for i, v in enumerate(auc):
        ax.text(i, v + 0.012, f"{v:.4f}".replace(".", ","), ha="center",
                va="bottom", fontsize=8, color=INK)
    ax.axhline(0.9752, color=GREY, linestyle="--", linewidth=0.9, zorder=4)
    ax.text(2.45, 0.9752, "taban 0,9752", fontsize=7, color=MUTED,
            va="bottom", ha="right")
    ax.set_ylim(0, 1.09)
    ax.set_title("Ham ROC-AUC", fontsize=9, color=INK, pad=8)

    ax = axes[1]
    ax.bar(range(3), head, color=cols, width=0.55, zorder=3)
    for i, v in enumerate(head):
        ax.text(i, v + 1.2, f"%{v:.1f}".replace(".", ","), ha="center",
                va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, 109)
    ax.set_title("Taban üstü açıklığın kapatılan payı", fontsize=9,
                 color=INK, pad=8)

    for ax in axes:
        ax.set_xticks(range(3))
        ax.set_xticklabels(lab, fontsize=7.5)
        tidy(ax)
    fig.savefig(f"{OUT}/sekil2_taban_ve_aciklik.png")
    plt.close(fig)


def fig3():
    """Fusion minus 2B, four datasets -- polarity."""
    lab = ["Özgün, genlik korunmuş\n(kapılı)",
           "Özgün, genlik korunmuş\n(doğrusal)",
           "Özgün, pencere bazlı norm.\n(doğrusal)",
           "Zor negatif\n(doğrusal)"]
    val = [-0.0034, -0.0049, -0.0102, 0.0026]
    cols = [ORANGE if v < 0 else BLUE for v in val]

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.bar(range(4), val, color=cols, width=0.55, zorder=3)
    for i, v in enumerate(val):
        off = 0.0006 if v > 0 else -0.0006
        ax.text(i, v + off, f"{v:+.4f}".replace(".", ","), ha="center",
                va="bottom" if v > 0 else "top", fontsize=8, color=INK)
    ax.axhline(0, color=INK, linewidth=1.0, zorder=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(lab, fontsize=7.5)
    ax.set_ylabel("Birleştirme − 2B (ROC-AUC)")
    ax.set_ylim(-0.0135, 0.0055)
    ax.set_title("Birleştirmenin katkısı, negatif seçimine göre işaret "
                 "değiştirmektedir", fontsize=9, color=INK, pad=10)
    tidy(ax)
    fig.savefig(f"{OUT}/sekil3_birlestirme_isaret.png")
    plt.close(fig)


def main():
    """Writes all three figures. Named so `sk figures` can dispatch to it."""
    fig1()
    fig2()
    fig3()
    print("wrote 3 figures to", OUT)
    return 0


if __name__ == "__main__":
    main()
