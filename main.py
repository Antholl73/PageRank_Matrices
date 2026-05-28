#====================================================
#Integrantes:
# Bernedo Coya Jose Daniel
# Avilés Fuentes Anthony Francisco
#Profesores:
# Daniel Alexis Gutierrez Pachas
# Rosmery Violeta Quispe Zavala
#Grupo:
# CComp 3-1
#====================================================

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# ===================================================
# CREACION E INICIALIZACION DEL GRAFO DIRIGIDO 'G'
# ===================================================

G = nx.DiGraph()

G.add_edges_from([
    (0,4),
    (1,4),
    (2,4),
    (3,4),
    (5,4),
    (6,4),
    (7,4),

    (0,1),
    (1,2),
    (2,3),

    (5,6),
    (6,7)
])

# ===================================================
# MATRIZ DE ADYACENCIA DEL GRAGO 'G'
# ===================================================

A = nx.to_numpy_array(G, dtype=float)

n = A.shape[0]

# ===================================================
# MATRIZ DE TRANSICION 'P'
# ===================================================

P = np.zeros((n,n))

for i in range(n):

    suma_fila = np.sum(A[i])

    if suma_fila != 0:
        P[i] = A[i] / suma_fila
    else:
        P[i] = 1 / n

# ===================================================
# MATRIZ DE GOOGLE 'G_google'
# ===================================================

alpha = 0.85

G_google = (
    alpha * P +
    (1 - alpha) * np.ones((n,n)) / n
)

# ===================================================
# AUTOVALORES Y AUTOVECTORES
# ===================================================
autovalores, autovectores = np.linalg.eig(G_google.T)

# Buscar autovalor más cercano a 1
indice = np.argmin(np.abs(autovalores - 1))

# Obtener autovector usando la variable 'indice'
v = np.real(autovectores[:, indice])

# Aplicando fórmula de e: e_i=v_i/(v_1+v_2+...+v_n);
e = v / np.sum(v)

# ===================================================
# DICCIONARIO CON LOS VALORES DEL VECTOR ESTACIONARIO
# ===================================================

pagerank_dict = {}
for i, nodo in enumerate(G.nodes()):
    pagerank_dict[nodo] = e[i]

# ===================================================
# IMPRIMIR RESULTADOS
# ===================================================

print("\nVector Resultante:\n")
print(e)

print("\nPageRank:\n")

for nodo in sorted(pagerank_dict.keys()):
    valor = pagerank_dict[nodo]
    porcentaje = valor * 100
    print(
        f"Nodo {nodo}: "
        f"{valor:.4f} "
        f"({porcentaje:.2f}%)"
    )

# ===================================================
# VISUALIZACIÓN
# ===================================================

pos = nx.spring_layout(G, seed=42, k=2, iterations=100)

sizes = [
    8000 * pagerank_dict[n]
    for n in G.nodes()
]

node_colors = [
    pagerank_dict[n]
    for n in G.nodes()
]

plt.figure(figsize=(10,7))

nx.draw_networkx_nodes(
    G, pos, node_size=sizes, node_color=node_colors, cmap=plt.cm.Pastel1,
)

nx.draw_networkx_edges(
    G, pos, arrows=True, arrowstyle='-|>', arrowsize=30, width=2, edge_color='gray', connectionstyle='arc3,rad=0.1'
)

labels = {
    n: f"Nodo {n}\nPR={pagerank_dict[n]:.3f}"
    for n in G.nodes()
}

label_pos = {}
for nodo, (x, y) in pos.items():
    label_pos[nodo] = (x, y + 0.08)

nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=12, font_color='black')

plt.title("Visualización de PageRank", fontsize=25, fontweight='bold')
plt.axis('off')
plt.savefig("pagerank.png", dpi=300)
plt.show()
