import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np

# 1. DATI DEI NODI (Distanze raddoppiate per fare spazio ai testi)
# Formato: [X, Y, 'Massa', 'Z', 'Simbolo', 'Emivita']
nodes = {
    'Th228': [0, 32, '228', '90', 'Th', '1.9 yr'],
    'Ra224': [0, 24, '224', '88', 'Ra', '3.6 days'],
    'Rn220': [0, 16, '220', '86', 'Rn', '55 s'],
    'Po216': [0, 8,  '216', '84', 'Po', '0.14 s'],
    'Pb212': [0, 0,  '212', '82', 'Pb', '10.6 h'],
    # Diramazione 
    'Bi212': [6, 4,  '212', '83', 'Bi', '61 min'],
    'Po212': [12, 8, '212', '84', 'Po', r'3$\times 10^{-7}$ s'],
    'Tl208': [6, -4, '208', '81', 'Tl', '3.1 min'],
    'Pb208': [12, 0,  '208', '82', 'Pb', 'stable']
}

# 2. COLLEGAMENTI
edges = [
    ('Th228', 'Ra224', 'alpha'),
    ('Ra224', 'Rn220', 'alpha'),
    ('Rn220', 'Po216', 'alpha'),
    ('Po216', 'Pb212', 'alpha'),
    ('Pb212', 'Bi212', 'beta'),
    ('Bi212', 'Po212', 'beta'),
    ('Bi212', 'Tl208', 'alpha'),
    ('Po212', 'Pb208', 'alpha'),
    ('Tl208', 'Pb208', 'beta')
]

# Impostazione figura
fig, ax = plt.subplots(figsize=(10, 16))
ax.set_aspect('equal') # FONDAMENTALE: Mantiene gli ottagoni regolari
ax.axis('off')

# Raggio notevolmente ingrandito per contenere bene il testo
R = 2.6 

# 3. DISEGNO FRECCE
for src, tgt, type_ in edges:
    x1, y1 = nodes[src][0], nodes[src][1]
    x2, y2 = nodes[tgt][0], nodes[tgt][1]
    
    dx, dy = x2 - x1, y2 - y1
    dist = np.hypot(dx, dy)
    
    # Calcolo per far partire/arrivare la freccia esattamente dal bordo
    gap = R + 0.3
    sx, sy = x1 + gap * dx/dist, y1 + gap * dy/dist
    ex, ey = x2 - gap * dx/dist, y2 - gap * dy/dist
    
    # Freccia
    ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(facecolor='black', edgecolor='black', width=3.5, 
                                headwidth=14, headlength=14, shrink=0))
    
    # Testi alfa/beta
    mx, my = (sx + ex)/2, (sy + ey)/2
    if type_ == 'alpha':
        ax.text(mx - 0.4, my, r"$\alpha$", fontsize=24, ha='right', va='center')
    else:
        ax.text(mx - 0.2, my + 0.4, r"$\beta^-$", fontsize=24, ha='right', va='bottom')

# 4. DISEGNO OTTAGONI E TESTI
for key, data in nodes.items():
    x, y, mass, z, sym, hl = data
    
    # Colore bordo: Rosso per Bismuto e Tallio, Nero per gli altri
    edge_color = '#e3242b' if key in ['Bi212', 'Tl208'] else 'black'
    
    # Ombra posteriore
    shadow = RegularPolygon((x+0.25, y-0.25), 8, radius=R, orientation=np.pi/8, 
                            facecolor='black', alpha=0.35)
    ax.add_patch(shadow)
    
    # Ottagono principale (Tutti grigi)
    octa = RegularPolygon((x, y), 8, radius=R, orientation=np.pi/8, 
                          facecolor='#D0D4D8', edgecolor=edge_color, lw=3.5)
    ax.add_patch(octa)
    
    # Testi INTERNI ben proporzionati
    ax.text(x - 0.9, y + 0.6, mass, fontsize=17, ha='right', va='center')
    ax.text(x - 0.9, y - 0.3, z, fontsize=17, ha='right', va='center')
    ax.text(x + 0.3, y + 0.2, sym, fontsize=36, weight='bold', ha='center', va='center')
    
    hl_font = 16 if key == 'Po212' else 16
    ax.text(x, y - 1.3, hl, fontsize=hl_font, ha='center', va='center')

# Margini per evitare tagli ai bordi
ax.set_xlim(-4, 16)
ax.set_ylim(-10, 36)

# Salvataggio diretto (così non devi preoccuparti di come si vede nella finestra di Python)
plt.tight_layout()
plt.savefig("figures/catena_decadimento_Th228.png", dpi=300, bbox_inches='tight')
plt.show()
print("Grafico finale salvato come catena_decadimento_Th228.png")