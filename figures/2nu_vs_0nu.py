import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13   # numbers on the axes (uncomment if needed)
plt.rcParams['ytick.labelsize'] = 13

RED, BLUE = '#fe2f33', '#004284'

def distribution_2nu(x, a=1, b=2, height=2.3):
    """Spettro schematico della somma delle energie degli elettroni (x = E/Q).
    a sposta il picco, b controlla la coda verso Q (b più basso -> coda più alta)."""
    dist = np.where(x <= 1.0, (x**a) * ((1.0 - x)**b), 0.0)
    return dist / dist.max() * height

def distribution_0nu(x, sigma=0.005):
    val_0nu = (np.exp(-(x-1)**2 / (2*sigma**2))) / (sigma * np.sqrt(2*np.pi))
    
    print(val_0nu/val_0nu.max())   
    return val_0nu/val_0nu.max()

# La curva 2nu arriva fino a Q (x = 1)
energy_2nu = np.linspace(0, 0.98, 10000)
pdf_2nu = distribution_2nu(energy_2nu, a=1, b=2)
energy_0nu = np.linspace(0.98, 1.02, 1000)
pdf_0nu = distribution_0nu(x=energy_0nu)

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(energy_2nu, pdf_2nu, color=BLUE, lw=3)
ax.plot(energy_0nu, pdf_0nu, color=RED, lw=3)

# 0nu: delta di Dirac a E = Q, disegnata come riga verticale alta 1 a.u.
# ax.vlines(1.0, 0, 1.0, color=RED, lw=3)

# Legenda con handle espliciti (vlines non genera una voce adatta da solo)
handles = [
    Line2D([0], [0], color=RED, lw=2, label=r"$0\nu\beta\beta$"),
    Line2D([0], [0], color=BLUE, lw=2, label=r"$2\nu\beta\beta$"),
]
ax.legend(handles=handles, labelspacing=1, fontsize=14, frameon=False, loc="upper right")

ax.set_xlim(0, 1.1)
ax.set_ylim(0, 2.5)
ax.set_xlabel(r"$(E_{e, 1} + E_{e, 2})$/$Q_{\beta\beta}$", fontsize=14)
ax.set_ylabel("a.u.", fontsize=14)
ax.set_title("Energy distribution comparison between double beta decay modes", fontsize=16)
ax.grid(alpha=0.3, ls='--')
fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/0nu_vs_2nu.pdf")
plt.show()