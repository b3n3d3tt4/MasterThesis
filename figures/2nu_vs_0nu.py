import os
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, sigma, mu):
    # Corretta la formula della gaussiana (rimosso il fattore errato e normalizzata l'altezza)
    g = np.exp(-(x - mu)**2 / (2 * sigma**2))
    return g

def distribution_2nu(x):
    dist = np.where(x <= 1.0, (x**2) * ((1.0 - x)**5), 0)
    max_val = np.max(dist)
    if max_val > 0:
        dist = dist / max_val * 2.85  # Riscalato all'altezza corretta rispetto al blu
    return dist

energy_2nu = np.linspace(0, 0.93, 10000)
energy_0nu = np.linspace(0.93, 1.06, 200)

pdf_2nu = distribution_2nu(energy_2nu)
pdf_0nu = gaussian(energy_0nu, sigma=0.02, mu=1.0)

plt.figure(figsize=(8, 6))

# Aggiunti i colori per rispettare lo schema della tesi
plt.plot(energy_0nu, pdf_0nu, label=r"$0\nu\beta\beta$", color='blue')
plt.plot(energy_2nu, pdf_2nu, label=r"$2\nu\beta\beta$", color='red')

plt.xlabel(r"E$_e$/Q$_{\beta\beta}$")
plt.ylabel("a.u.")
plt.title("Energy distribution comparison between double beta decay modes")
plt.legend(labelspacing=1, fontsize=12)
plt.grid(alpha=0.25)
plt.tight_layout()

# Evita errori se la cartella figures non esiste
os.makedirs("figures", exist_ok=True)

plt.savefig("figures/0nu_vs_2nu.pdf", dpi=300)
plt.show()