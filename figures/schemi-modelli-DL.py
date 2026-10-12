"""
Schemi con matplotlib, salvati in PDF (e in PNG):
  1) VAE.pdf             encoder -> (mu, sigma) -> reparametrisation trick -> z -> decoder
  2) VAE_classifier.pdf  encoder del VAE + classificatore MLP (32 -> 16 -> 1) sulle medie mu
  3) TAE.pdf             Temporal AutoEncoder: il latente e' ancora una serie temporale
  4) TAE_clustering.pdf  Deep Temporal Clustering: TAE + Time Series Clustering

Per le scritte "a mano" serve il font Kalam (Google Fonts), nella stessa cartella dello script:
  https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Regular.ttf
Se il file non c'e', viene usato un font corsivo di ripiego.

Tutte le coordinate sono in "pixel" con l'origine in alto a sinistra (y verso il basso).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrowPatch, FancyBboxPatch

# =============================================================================
# STILE
# =============================================================================
NAVY, ENC_FILL = "#0F2D52", "#E3E8EE"
ORANGE, ORANGE_FILL = "#F5A030", "#FFF3E6"
RED, RED_FILL = "#E8434F", "#FDE8EA"
GRAY, GRAY_FILL = "#8A8F98", "#F1F2F4"
BLUE_HAND = "#1F5F99"
NODE_FILL, NODE_EDGE, EDGE_GRAY = "#E2EEF5", "#6FA6C8", "#B5B8C0"

# Percorsi relativi alla cartella dello script, non alla cartella da cui viene lanciato
# (VS Code di solito lancia dalla cartella del workspace)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "schemi_output")
FONT_FILE = os.path.join(SCRIPT_DIR, "Kalam-Regular.ttf")
if os.path.exists(FONT_FILE):
    font_manager.fontManager.addfont(FONT_FILE)
    # DejaVu Sans fa da ripiego per i glifi che Kalam non ha (lettere greche, simboli)
    HAND = [font_manager.FontProperties(fname=FONT_FILE).get_name(), "DejaVu Sans"]
else:
    HAND = ["cursive", "DejaVu Sans"]
    print(f"ATTENZIONE: font non trovato in {FONT_FILE}, uso un font di ripiego")

SANS = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "stix"


# =============================================================================
# FUNZIONI DI SUPPORTO
# =============================================================================
def new_canvas(width, height, scale=0.006, y0=0):
    """Figura con asse in coordinate 'pixel', origine in alto a sinistra."""
    fig, ax = plt.subplots(figsize=(width * scale, height * scale))
    ax.set_xlim(0, width)
    ax.set_ylim(y0 + height, y0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def save(fig, out):
    """Salva in PDF (vettoriale) e in PNG (200 dpi) con lo stesso nome."""
    os.makedirs(OUT_DIR, exist_ok=True)
    name = os.path.join(OUT_DIR, os.path.splitext(out)[0])
    fig.savefig(name + ".pdf", facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(name + ".png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"salvato: {name}.pdf")


def arrow(ax, x0, y0, x1, y1, lw=2.2, color="black", ms=22, ls="-", z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=ms,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0,
                                 linestyle=ls, zorder=z))


def line(ax, xs, ys, lw=2.2, color="black", ls="-", z=2):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt",
            solid_joinstyle="miter", zorder=z)


def label_box(ax, x0, y0, x1, y1, text, fs, lw=2.5):
    """Riquadro con testo verticale in maiuscolo, centrato sull'altezza delle maiuscole."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="black", lw=lw))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    # conversione punti tipografici -> unita' dati
    px_per_unit = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    pt = ax.figure.dpi / 72 / px_per_unit
    cap = 0.73 * fs * pt     # altezza delle maiuscole (DejaVu Sans Bold)
    step = 1.15 * fs * pt    # distanza tra le righe

    lines = text.split("\n")
    width = (len(lines) - 1) * step + cap
    for i, ln in enumerate(lines):
        xb = cx - width / 2 + cap + i * step   # baseline della riga i (testo ruotato di 90 gradi)
        ax.text(xb, cy, ln, rotation=90, rotation_mode="anchor", ha="center", va="baseline",
                fontsize=fs, fontweight="bold", family=SANS)


def trapezoid(ax, pts, text, fs, lw=2.5):
    ax.add_patch(Polygon(pts, closed=True, fc=ENC_FILL, ec=NAVY, lw=lw))
    ax.text(sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4, text, ha="center",
            va="center", fontsize=fs, fontweight="bold", color=NAVY, family=SANS)


def latent_box(ax, x0, y0, x1, y1, symbol, fc, ec, fs, lw=2.5, ls="-", alpha=1):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=lw, ls=ls,
                           alpha=alpha, zorder=3))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, symbol, ha="center", va="center",
            fontsize=fs, color=ec, alpha=alpha, zorder=4)


def times8(ax, x, y, color, fs, alpha=1):
    ax.text(x, y, "x8", ha="center", va="center", fontsize=fs, fontweight="bold",
            color=color, family=SANS, alpha=alpha)


def op_node(ax, x, y, symbol, r=28):
    """Nodo circolare per le operazioni (+, x)."""
    ax.add_patch(Circle((x, y), r, fc="white", ec="black", lw=2.2, zorder=4))
    ax.text(x, y - 1, symbol, ha="center", va="center", fontsize=24, zorder=5)


def hand(ax, x, y, s, fs=14, color=BLUE_HAND, ha="center", va="center", ls=1.1):
    """Testo in stile manoscritto."""
    ax.text(x, y, s, family=HAND, color=color, fontsize=fs, ha=ha, va=va, linespacing=ls)


def dots(ax, x, y0, y1, n=3):
    """Puntini verticali tra i neuroni."""
    ax.scatter([x] * n, np.linspace(y0, y1, n), s=7, color=BLUE_HAND, zorder=3)


# =============================================================================
# 1) VAE
# =============================================================================
def schema_vae(out="VAE.pdf"):
    W, H, Y = 2680, 1150, 520          # Y = asse orizzontale principale
    fig, ax = new_canvas(W, H, y0=-140)

    label_box(ax, 38, 230, 135, 810, "ORIGINAL WAVEFORM", fs=17)
    arrow(ax, 180, Y, 325, Y)
    trapezoid(ax, [(373, Y - 190), (758, Y - 95), (758, Y + 95), (373, Y + 190)],
              "ENCODER", fs=19)

    # ---------------- mu e sigma
    Ym, Ys = 360, 680
    line(ax, [806, 903], [Y, Y])
    line(ax, [903, 903], [Ym, Ys])
    arrow(ax, 903, Ym, 952, Ym)
    arrow(ax, 903, Ys, 952, Ys)

    latent_box(ax, 999, Ym - 70, 1139, Ym + 70, r"$\mu$", ORANGE_FILL, ORANGE, 48)
    times8(ax, 1069, Ym + 95, ORANGE, 15)
    latent_box(ax, 999, Ys - 70, 1139, Ys + 70, r"$\sigma$", ORANGE_FILL, ORANGE, 48)
    times8(ax, 1069, Ys - 95, ORANGE, 15)

    # ---------------- la distribuzione q(z|x) = N(mu, sigma^2)
    ax.add_patch(FancyBboxPatch((985, 28), 510, 215, boxstyle="round,pad=0,rounding_size=18",
                                fc="#FFFAF3", ec=ORANGE, lw=1.6, ls="--", zorder=1))
    m, s, base, hgt = 1240, 62, 212, 140
    xs = np.linspace(1015, 1465, 300)
    ys = base - hgt * np.exp(-(xs - m) ** 2 / (2 * s ** 2))
    m0, s0 = 1180, 95                  # prior N(0, 1), piu' largo e centrato altrove
    ax.plot(xs, base - 0.65 * hgt * np.exp(-(xs - m0) ** 2 / (2 * s0 ** 2)), color=GRAY,
            lw=1.6, ls="--", zorder=2)
    hand(ax, 1040, 128, "N(0, 1)", fs=11, color=GRAY)
    ax.fill_between(xs, ys, base, color=ORANGE, alpha=0.18, lw=0, zorder=2)
    ax.plot(xs, ys, color=ORANGE, lw=2.4, zorder=3)
    line(ax, [1015, 1465], [base, base], lw=1.4, color="#555", z=3)
    line(ax, [m, m], [base, base - hgt], lw=1.4, color=ORANGE, ls="--", z=3)

    yh = base - hgt * np.exp(-0.5)     # altezza a un sigma dal centro
    ax.annotate("", (m + s, yh), (m, yh),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6, shrinkA=0, shrinkB=0),
                zorder=4)
    ax.text(m + s / 2, yh + 17, r"$\sigma$", ha="center", va="center", fontsize=20, color=ORANGE)
    ax.text(m, base + 16, r"$\mu$", ha="center", va="center", fontsize=20, color=ORANGE)

    zs = m + 1.25 * s                  # un campione z estratto dalla gaussiana
    ax.add_patch(Circle((zs, base), 8, fc=RED, ec=RED, zorder=5))
    ax.text(zs, base + 17, r"$z$", ha="center", va="center", fontsize=20, color=RED)

    hand(ax, 1000, 52, "q(z | x) = N(μ, σ²)", fs=13, color="#C47A12", ha="left")
    line(ax, [1069, 1069], [243, Ym - 70], lw=1.6, color=ORANGE, ls="--", z=1)

    # ---------------- reparametrisation trick: z = mu + sigma x epsilon
    xo = 1290                          # nodo x
    op_node(ax, xo, Ys, "×")
    arrow(ax, 1139, Ys, xo - 28, Ys)

    latent_box(ax, xo - 70, 850, xo + 70, 990, r"$\varepsilon$", GRAY_FILL, GRAY, 44, ls="--")
    times8(ax, xo - 100, 920, GRAY, 15)
    arrow(ax, xo, 850, xo, Ys + 28)
    hand(ax, xo + 92, 920,
         "ε ~ N(0, I)\n"
         "random noise, sampled outside\n"
         "the network: no parameters\n"
         "to learn, no gradient needed",
         fs=11, ha="left", color="#555")

    xp = 1460                          # nodo +
    op_node(ax, xp, Y, "+")
    line(ax, [1139, xp], [Ym, Ym])
    arrow(ax, xp, Ym, xp, Y - 28)
    line(ax, [xo + 28, xp], [Ys, Ys])
    arrow(ax, xp, Ys, xp, Y + 28)

    arrow(ax, xp + 28, Y, 1540, Y)
    latent_box(ax, 1540, Y - 70, 1680, Y + 70, r"$z$", RED_FILL, RED, 48)
    times8(ax, 1610, Y + 95, RED, 15)
    ax.text(1635, Y - 98, r"$z = \mu + \sigma \times \varepsilon$", ha="center", va="center",
            fontsize=20, color=RED)

    # ---------------- backprop: gradienti tratteggiati, paralleli al percorso in avanti
    g = dict(lw=1.5, color=RED, ls=(0, (4, 3)), z=1)
    line(ax, [xp + 24, xp + 24, 1160], [Y - 38, Ym - 24, Ym - 24], **g)
    arrow(ax, 1175, Ym - 24, 1146, Ym - 24, lw=1.5, color=RED, ms=14)
    line(ax, [xp - 24, xp - 24, 1160], [Y + 38, Ys - 40, Ys - 40], **g)
    arrow(ax, 1175, Ys - 40, 1146, Ys - 40, lw=1.5, color=RED, ms=14)
    ax.text(1300, Ym - 50, r"$\partial \mathcal{L} / \partial \mu$", ha="center", va="center",
            fontsize=17, color=RED)
    ax.text(1300, Ys - 66, r"$\partial \mathcal{L} / \partial \sigma$", ha="center", va="center",
            fontsize=17, color=RED)

    hand(ax, 1505, Y + 135,
         "the gradient of the loss (dashed) flows\n"
         "back through + and × to μ, σ and the\n"
         "encoder, while all the randomness is in ε",
         fs=12, ha="left", va="top", color=RED)

    # ---------------- decoder
    arrow(ax, 1720, Y, 1860, Y)
    trapezoid(ax, [(1907, Y - 95), (2292, Y - 190), (2292, Y + 190), (1907, Y + 95)],
              "DECODER", fs=19)
    arrow(ax, 2338, Y, 2483, Y)
    label_box(ax, 2530, 230, 2627, 810, "RECONSTRUCTED\nWAVEFORM", fs=15)

    # ---------------- loss: ricostruzione (NLL) + KL sullo spazio latente
    yn = -45                            # parentesi rossa tra originale e ricostruita
    line(ax, [86.5, 86.5, 2578.5, 2578.5], [230, yn, yn, 230], lw=1.6, color=RED)
    ax.text(1240, yn - 32, r"$\mathcal{L}_{\mathrm{NLL}}(x, \hat{x})$", ha="center",
            va="center", fontsize=21, color=RED)

    ax.text(1530, 105,
            r"$\mathcal{L}_{\mathrm{KL}} = \sum_j \mathrm{KL}\left(\,\mathcal{N}(\mu_j, \sigma_j^2)"
            r"\;\|\;\mathcal{N}(0, 1)\,\right)$",
            ha="left", va="center", fontsize=19, color=RED)
    hand(ax, 1530, 165, "pulls each latent Gaussian towards the prior N(0, 1)",
         fs=12, ha="left", color=RED)

    ax.text(2290, 930, r"$\mathcal{L} = \mathcal{L}_{\mathrm{NLL}} + \beta\, \mathcal{L}_{\mathrm{KL}}$",
            ha="center", va="center", fontsize=23, color=RED)

    save(fig, out)


# =============================================================================
# 2) ENCODER + CLASSIFICATORE
# =============================================================================
def schema_classificatore(out="VAE_classifier.pdf"):
    W, H, Y = 1720, 500, 380
    fig, ax = new_canvas(W, H, scale=0.009, y0=Y - 245)

    label_box(ax, 42, Y - 180, 102, Y + 180, "ORIGINAL WAVEFORM", fs=14, lw=2)
    arrow(ax, 130, Y, 220, Y, lw=1.8, ms=18)
    trapezoid(ax, [(249, Y - 120), (487, Y - 60), (487, Y + 60), (249, Y + 120)],
              "ENCODER", fs=15, lw=2)

    # ---------------- mu e sigma (sigma sbiadito: il classificatore usa solo mu)
    Ym, Ys = Y - 125, Y + 125
    line(ax, [517, 577], [Y, Y], lw=1.8)
    line(ax, [577, 577], [Ym, Ys], lw=1.8)
    arrow(ax, 577, Ym, 607, Ym, lw=1.8, ms=16)
    arrow(ax, 577, Ys, 607, Ys, lw=1.8, ms=16)

    latent_box(ax, 636, Ym - 45, 726, Ym + 45, r"$\mu$", ORANGE_FILL, ORANGE, 36, lw=2)
    times8(ax, 681, Ym + 65, ORANGE, 14)
    latent_box(ax, 636, Ys - 45, 726, Ys + 45, r"$\sigma$", ORANGE_FILL, ORANGE, 36, lw=2,
               alpha=0.35)
    times8(ax, 681, Ys + 65, ORANGE, 14, alpha=0.35)
    hand(ax, 681, Ys + 105, "not used by\nthe classifier", fs=11, color="#888")

    # mu -> rete (gomito verso il centro)
    line(ax, [746, 775, 775], [Ym, Ym, Y], lw=1.8)
    arrow(ax, 775, Y, 815, Y, lw=1.8, ms=16)

    # ---------------- rete: per ogni layer un neurone centrale, puntini, uno sopra e uno sotto
    r = 15
    x1, x2, x3 = 850, 1000, 1150
    L1 = [(x1, Y - 150), (x1, Y), (x1, Y + 150)]
    L2 = [(x2, Y - 90), (x2, Y), (x2, Y + 90)]
    L3 = [(x3, Y)]

    for a in L1:
        for b in L2:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=EDGE_GRAY, lw=0.9, zorder=1)
    for a in L2:
        ax.plot([a[0], x3], [a[1], Y], color=EDGE_GRAY, lw=0.9, zorder=1)
    for (x, y) in L1 + L2 + L3:
        ax.add_patch(Circle((x, y), r, fc=NODE_FILL, ec=NODE_EDGE, lw=1.2, zorder=2))

    dots(ax, x1, Y - 112, Y - 38, n=4)
    dots(ax, x1, Y + 38, Y + 112, n=4)
    dots(ax, x2, Y - 62, Y - 28)
    dots(ax, x2, Y + 28, Y + 62)

    hand(ax, x1, Y + 205, "32\nneurons")
    hand(ax, x2, Y + 145, "16\nneurons")
    hand(ax, x3, Y + 55, "1\nneuron")

    # ---------------- attivazioni: freccia da un neurone a quello del layer successivo
    def act_arrow(p, q, text, text_y):
        ax.add_patch(FancyArrowPatch((p[0] + 5, p[1] - 15), (q[0] - 5, q[1] - 15),
                                     connectionstyle="arc3,rad=-0.45", arrowstyle="->",
                                     mutation_scale=13, lw=1.3, color=NAVY, zorder=4))
        hand(ax, (p[0] + q[0]) / 2, text_y, text, fs=10, va="bottom")

    act_arrow(L1[0], L2[0], "LeakyReLU with α = 0.1\n+ Dropout (0.2)", Y - 180)
    act_arrow(L2[0], L3[0], "LeakyReLU\nwith α = 0.1", Y - 118)

    # ---------------- output
    arrow(ax, 1188, Y, 1250, Y, lw=1.8, ms=16)
    hand(ax, 1265, Y,
         "p = sigmoid(output) ∈ [0, 1]\n"
         "which is the probability\n"
         "of belonging to class 1\n"
         "(i.e. being a MSE signal)",
         fs=12, ha="left", ls=1.05)

    # ---------------- loss: BCEWithLogitsLoss (la sigmoide e' applicata dentro la loss)
    ax.text(1265, Y + 120,
            r"$\mathcal{L}_{\mathrm{BCE}} = -\left[\,y \log p + (1 - y) \log (1 - p)\,\right]$",
            ha="left", va="center", fontsize=15, color=RED)
    hand(ax, 1265, Y + 145,
         "BCEWithLogitsLoss: y = true label,\nthe sigmoid is applied inside the loss",
         fs=11, ha="left", va="top", color=RED)

    save(fig, out)


# =============================================================================
# 3) e 4) DEEP TEMPORAL CLUSTERING
# =============================================================================
C1, C1_FILL = "#2E6FB7", "#E6EFF9"     # cluster 1
C2, C2_FILL = "#2E9E6B", "#E5F4EC"     # cluster 2
AXIS_GRAY = "#9AA0A8"
T_LATENT = 20                          # 200 campioni / pool_size 10


# =============================================================================
# SERIE TEMPORALI (casuali ma lisce, con seed fisso)
# =============================================================================
def random_series(seed, n=T_LATENT, phi=0.4):
    """Serie casuale irregolare (processo AR(1)): nessun picco o forma particolare."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x


Z = random_series(5)                                                    # latente
W1 = Z + np.random.default_rng(11).normal(0, 0.45, T_LATENT)             # centroide vicino a z
W2 = random_series(42) + 3.2                                            # centroide diverso


# =============================================================================
# FUNZIONI DI SUPPORTO
# =============================================================================
def series_box(ax, x0, y0, x1, y1, series, colors, ec, fc, labels=None, lw=2.5):
    """Riquadro con una o piu' serie temporali disegnate dentro, con asse dei tempi."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=lw, zorder=3))

    # asse dei tempi
    ya = y1 - 30
    arrow(ax, x0 + 18, ya, x1 - 16, ya, lw=1.2, color=AXIS_GRAY, ms=10, z=4)
    hand(ax, x1 - 30, ya + 15, "t", fs=12, color=AXIS_GRAY)

    allv = np.concatenate(series)
    lo, hi = allv.min(), allv.max()
    t = np.linspace(x0 + 30, x1 - 30, len(series[0]))
    ys = [(y1 - 48) - (s - lo) / (hi - lo) * (y1 - y0 - 90) for s in series]
    for y, c in zip(ys, colors):
        ax.plot(t, y, color=c, lw=2, zorder=5)
        ax.scatter(t, y, s=14, color=c, zorder=6)
    if labels:
        # etichette fuori dal riquadro, a sinistra, all'altezza del primo punto di ogni curva
        for y, c, lab in zip(ys, colors, labels):
            ax.text(x0 - 10, y[0], lab, ha="right", va="center", fontsize=17, color=c,
                    zorder=6)


def bars_box(ax, x0, y0, x1, y1, values, colors, ec="#555", fc="white"):
    """Riquadro con un istogramma delle probabilita' per cluster."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=2, zorder=3))
    base, hmax, bw = y1 - 38, y1 - y0 - 80, 46
    line(ax, [x0 + 20, x1 - 20], [base, base], lw=1.2, color=AXIS_GRAY, z=4)
    xs = np.linspace(x0, x1, len(values) + 2)[1:-1]
    for k, (x, v, c) in enumerate(zip(xs, values, colors)):
        ax.add_patch(Rectangle((x - bw / 2, base - v * hmax), bw, v * hmax, fc=c, ec=c,
                               alpha=0.85, zorder=5))
        hand(ax, x, base - v * hmax - 14, f"{v:.2f}", fs=10, color=c)
        hand(ax, x, base + 17, f"j = {k + 1}", fs=10, color="#555")


def block(ax, x0, y0, x1, y1, title, subtitle=None):
    """Blocco operativo a bordi arrotondati, nello stile di encoder/decoder."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=16",
                                fc=ENC_FILL, ec=NAVY, lw=2.2, zorder=3))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if subtitle:
        ax.text(cx, cy - 16, title, ha="center", va="center", fontsize=14, fontweight="bold",
                color=NAVY, family=SANS, zorder=4)
        ax.text(cx, cy + 22, subtitle, ha="center", va="center", fontsize=18, color=NAVY,
                zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=14, fontweight="bold",
                color=NAVY, family=SANS, zorder=4, linespacing=1.15)


def tae_row(ax, Y):
    """Riga del Temporal AutoEncoder; restituisce le coordinate del riquadro latente."""
    label_box(ax, 38, Y - 290, 135, Y + 290, "ORIGINAL WAVEFORM", fs=17)
    arrow(ax, 180, Y, 325, Y)
    trapezoid(ax, [(373, Y - 190), (758, Y - 95), (758, Y + 95), (373, Y + 190)],
              "TEMPORAL\nENCODER", fs=19)
    arrow(ax, 806, Y, 950, Y)

    lat = (1000, Y - 110, 1380, Y + 110)
    series_box(ax, *lat, [Z], [RED], RED, RED_FILL)
    ax.text((lat[0] + lat[2]) / 2, lat[1] - 32, r"$z(t)$", ha="center", va="center",
            fontsize=24, color=RED)
    hand(ax, (lat[0] + lat[2]) / 2, lat[3] + 30, "latent time series: 20 time steps × 1 feature",
         fs=12, color=RED)

    arrow(ax, 1430, Y, 1575, Y)
    trapezoid(ax, [(1623, Y - 95), (2008, Y - 190), (2008, Y + 190), (1623, Y + 95)],
              "TEMPORAL\nDECODER", fs=19)
    arrow(ax, 2056, Y, 2200, Y)
    label_box(ax, 2248, Y - 290, 2345, Y + 290, "RECONSTRUCTED\nWAVEFORM", fs=15)
    return lat


# =============================================================================
# 1) TEMPORAL AUTOENCODER
# =============================================================================
def nll_bracket(ax, Y, y_line, x_label):
    """Parentesi rossa tra waveform originale e ricostruita, con la perdita NLL."""
    line(ax, [86.5, 86.5, 2296.5, 2296.5], [Y - 290, y_line, y_line, Y - 290], lw=1.6, color=RED)
    ax.text(x_label, y_line - 30, r"$\mathcal{L}_{\mathrm{NLL}}(x, \hat{x})$", ha="center",
            va="center", fontsize=21, color=RED)


def schema_tae(out="TAE.pdf"):
    W, H, Y = 2380, 760, 440
    fig, ax = new_canvas(W, H)
    lat = tae_row(ax, Y)
    nll_bracket(ax, Y, Y - 345, (lat[0] + lat[2]) / 2)
    save(fig, out)


# =============================================================================
# 2) DEEP TEMPORAL CLUSTERING (TAE + clustering sotto)
# =============================================================================
def panel(ax, x0, y0, x1, y1, title):
    """Riquadro tratteggiato che raggruppa un blocco del modello, con titolo in alto a sinistra."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=22",
                                fc="#FAFBFC", ec=NAVY, lw=1.6, ls="--", zorder=0))
    ax.text(x0 + 25, y0 + 28, title, ha="left", va="center", fontsize=15,
            fontweight="bold", color=NAVY, family=SANS)


def schema_dtc(out="TAE_clustering.pdf"):
    W, H, Y = 2540, 1420, 470
    BY = Y + 600                        # asse della parte di clustering
    fig, ax = new_canvas(W, H)

    # ---------------- TEMPORAL AUTOENCODER (riquadro tratteggiato + perdita NLL)
    tae_top, tae_bot = Y - 335, Y + 290
    panel(ax, 345, tae_top, 2035, tae_bot, "TEMPORAL AUTOENCODER")
    lat = tae_row(ax, Y)
    xl = (lat[0] + lat[2]) / 2

    nll_bracket(ax, Y, tae_top - 50, xl)

    # ---------------- TIME SERIES CLUSTERING
    tsc_top = BY - 200
    panel(ax, 520, tsc_top, 2510, BY + 300, "TIME SERIES CLUSTERING")
    arrow(ax, xl, lat[3] + 55, xl, BY - 70, color=RED)   # il latente va al blocco della distanza

    # centroidi (anch'essi serie temporali)
    cen = (580, BY - 110, 940, BY + 110)
    series_box(ax, *cen, [W1, W2], [C1, C2], NAVY, "white",
               labels=[r"$w_1$", r"$w_2$"], lw=2)
    hand(ax, (cen[0] + cen[2]) / 2, cen[3] + 45, "cluster centroids\ninitialised with KMeans",
         fs=12)

    # distanza
    dist = (1040, BY - 70, 1340, BY + 70)
    block(ax, *dist, "DISTANCE", r"$d(z_i, w_j)$")
    arrow(ax, cen[2], BY, dist[0], BY)
    hand(ax, (dist[0] + dist[2]) / 2, dist[3] + 30, "euclidean, CID or ACF", fs=12)

    # kernel t di Student, con la normalizzazione al denominatore
    stu = (1430, BY - 70, 1730, BY + 70)
    block(ax, *stu, "STUDENT'S t\nKERNEL")
    arrow(ax, dist[2], BY, stu[0], BY)
    ax.text((stu[0] + stu[2]) / 2, stu[3] + 115,
            r"$q_{ij} = \dfrac{\left(1 + \dfrac{d(z_i, w_j)^2}{\alpha}\right)^{-\frac{\alpha+1}{2}}}"
            r"{\sum_k \left(1 + \dfrac{d(z_i, w_k)^2}{\alpha}\right)^{-\frac{\alpha+1}{2}}}$",
            ha="center", va="center", fontsize=16, color=NAVY)

    # assegnazioni soft q e distribuzione bersaglio p
    qb = (1830, BY - 110, 2080, BY + 110)
    pb = (2220, BY - 110, 2470, BY + 110)
    arrow(ax, stu[2], BY, qb[0], BY)
    bars_box(ax, *qb, [0.78, 0.22], [C1, C2])
    bars_box(ax, *pb, [0.93, 0.07], [C1, C2])
    for box, sym, txt in [(qb, r"$q_{ij}$", "soft assignments"),
                          (pb, r"$p_{ij}$", "target distribution")]:
        ax.text((box[0] + box[2]) / 2, box[1] - 28, sym, ha="center", va="center",
                fontsize=22, color=NAVY)
        hand(ax, (box[0] + box[2]) / 2, box[1] - 64, txt, fs=12)
    arrow(ax, qb[2], BY, pb[0], BY, lw=1.8, ms=16)
    ax.text((qb[2] + pb[0]) / 2, BY - 42, r"$p_{ij} \propto \frac{q_{ij}^{\,2}}{f_j}$",
            ha="center", va="center", fontsize=16, color=NAVY)

    # perdita KL(P || Q)
    yk = qb[3] + 42
    line(ax, [(qb[0] + qb[2]) / 2] * 2 + [(pb[0] + pb[2]) / 2] * 2,
         [qb[3], yk, yk, pb[3]], lw=1.6, color=RED)
    ax.text((qb[2] + pb[0]) / 2, yk + 32, r"$\mathcal{L}_{\mathrm{KL}}(P \,\|\, Q)$", ha="center",
            va="center", fontsize=19, color=RED)

    # perdita totale, nello spazio libero a sinistra del clustering
    ax.text(265, BY + 50,
            r"$\mathcal{L} = \mathcal{L}_{\mathrm{NLL}} + \gamma\, \mathcal{L}_{\mathrm{KL}}$",
            ha="center", va="center", fontsize=23, color=RED)

    save(fig, out)


if __name__ == "__main__":
    schema_vae()
    schema_classificatore()
    schema_tae()
    schema_dtc()
    print(f"\nTutti gli schemi sono in: {OUT_DIR}")