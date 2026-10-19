import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

BLUE, YELLOW, RED = "#004284", "#ffbf00", "#ff0000"

A = 76
# Dati completi (servono solo per il fit delle parabole guida)
MASS_EXCESS_A76 = {
    30: -62.042888, 31: -66.202888, 32: -73.212888, 33: -72.289575,
    34: -75.251894, 35: -70.288670, 36: -68.978700, 37: -60.480547,
    38: -54.380547,
}
# Isobari effettivamente mostrati
Z_SHOWN = [31, 32, 33, 34, 35]
SYMBOL = {30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br",
          36: "Kr", 37: "Rb", 38: "Sr"}


def plot_mass_parabola(out_path):
    shown = {z: MASS_EXCESS_A76[z] for z in Z_SHOWN}
    fig, ax = plt.subplots(figsize=(9, 6))

    # --- parabole guida (fit su tutti i dati, disegnate solo nel range) ---
    zs_all = np.array(sorted(MASS_EXCESS_A76))
    ee = np.array([z for z in zs_all if z % 2 == 0])
    oo = np.array([z for z in zs_all if z % 2 == 1])
    z_plot = np.linspace(min(Z_SHOWN) - 0.5, max(Z_SHOWN) + 0.5, 300)
    for zz, ls, lw in [(ee, "-", 1.5), (oo, ":", 2)]:
        c = np.polyfit(zz, [MASS_EXCESS_A76[z] for z in zz], 2)
        ax.plot(z_plot, np.polyval(c, z_plot), color=BLUE, ls=ls, lw=lw, zorder=1)

    # --- punti ---
    for Z, M in shown.items():
        ax.scatter(Z, M, s=60, color=BLUE, zorder=3, edgecolor="black", linewidth=0.6)
        even = Z % 2 == 0
        ax.annotate(rf"$^{{{A}}}${SYMBOL[Z]}", (Z, M), textcoords="offset points",
                    xytext=(0, -11 if even else 11), ha="center",
                    va="top" if even else "bottom", fontsize=10)

    # --- assi e layout: vanno fissati PRIMA di disegnare le frecce ---
    ax.set_xticks(Z_SHOWN)
    ax.set_ylim(-77.5, -62.0)
    ax.set_xlim(min(Z_SHOWN) - 0.5, max(Z_SHOWN) + 0.5)
    fig.tight_layout()

    # --- frecce: fusto (eventualmente tratteggiato) + punta sempre piena ---
    def arrow(z1, z2, color, dashed=False, lw=2, shrink_pt=8, head_pt=10):
        """Freccia dritta dall'isobaro z1 all'isobaro z2.
        shrink_pt : distanza (in punti) lasciata libera attorno ai marker
        head_pt   : lunghezza della punta (in punti)"""
        to_px = fig.dpi / 72.0
        p1 = ax.transData.transform((z1, shown[z1]))
        p2 = ax.transData.transform((z2, shown[z2]))
        u = (p2 - p1) / np.linalg.norm(p2 - p1)
        start = p1 + u * shrink_pt * to_px          # inizio fusto
        tip = p2 - u * shrink_pt * to_px            # vertice della punta
        base = tip - u * head_pt * to_px            # base punta = fine fusto
        inv = ax.transData.inverted()
        start, base, tip = (inv.transform(p) for p in (start, base, tip))

        # fusto, senza punta
        line, = ax.plot([start[0], base[0]], [start[1], base[1]], color=color,
                        lw=lw, solid_capstyle="butt", dash_capstyle="butt", zorder=2)
        if dashed:
            line.set_dashes((4, 2.5))
        # punta piena
        ax.add_patch(FancyArrowPatch(
            tuple(base), tuple(tip),
            arrowstyle="-|>,head_length=0.5,head_width=0.25",
            mutation_scale=head_pt * 2, color=color, lw=0,
            shrinkA=0, shrinkB=0, zorder=2))

    arrow(32, 33, YELLOW)                           # Ge -> As   beta singolo
    arrow(32, 34, RED, dashed=True, lw=2.5)         # Ge -> Se   doppio beta

    ax.set_xlabel("Atomic number $Z$", fontsize=12)
    ax.set_ylabel("Mass excess (MeV)", fontsize=12)
    ax.set_title(rf"Isobaric mass parabolas for $A = {A}$", fontsize=13)
    ax.grid(alpha=0.3, ls="--")

    handles = [
        Line2D([0], [0], color=BLUE, lw=1.5, label="even-even"),
        Line2D([0], [0], color=BLUE, lw=2, ls=":", label="odd-odd"),
        Line2D([0], [0], color=YELLOW, lw=2, label=r"single $\beta$ decay"),
        Line2D([0], [0], color=RED, lw=2.2, ls="--", label=r"double $\beta$ decay"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9)

    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    plot_mass_parabola(
        "C:\\Users\\Utente\\Desktop\\uni\\MAGISTRALE\\___TESI_magistrale\\latex\\figures\\mass_parabola_A76.png"
    )