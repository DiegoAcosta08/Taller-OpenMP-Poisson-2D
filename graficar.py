# =============================================================================
# Visualización de la solución de ∇²V = 4  —  Ejemplo 3
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os

# -----------------------------------------------------------------------
# Leer el archivo .dat dinámicamente
# -----------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Error: Debes proporcionar el archivo .dat como argumento.")
    print("Uso: python3 graficar.py data/solucion_algo.dat")
    sys.exit(1)

filename = sys.argv[1]

if not os.path.exists(filename):
    print(f"Error: no se encontró el archivo '{filename}'")
    sys.exit(1)

print(f"Leyendo {filename}...")
data = np.loadtxt(filename, comments="#")

x     = data[:, 0]
y     = data[:, 1]
V_num = data[:, 2]
V_exa = data[:, 3]
error = data[:, 4]

# Extraer el nombre base para los archivos de salida (ej. de "data/solucion_task.dat" saca "task")
base_name = os.path.basename(filename) # "solucion_task.dat"
name_without_ext = os.path.splitext(base_name)[0] # "solucion_task"
method_name = name_without_ext.replace("solucion_", "") # "task"

# -----------------------------------------------------------------------
# Reconstruir grilla 2D
# -----------------------------------------------------------------------
x_vals = np.unique(x)
y_vals = np.unique(y)
M = len(x_vals) - 1
N = len(y_vals) - 1

X = x.reshape(M + 1, N + 1)
Y = y.reshape(M + 1, N + 1)
Z_num = V_num.reshape(M + 1, N + 1)
Z_exa = V_exa.reshape(M + 1, N + 1)
Z_err = error.reshape(M + 1, N + 1)

print(f"Malla: {M} x {N}  |  Error máximo: {error.max():.2e}")

# -----------------------------------------------------------------------
# Figura 1: comparación numérica vs analítica
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(f"Poisson 2D ({method_name}) — $\\nabla^2 V = 4$,  $V_{{exact}}=(x-y)^2$", fontsize=13)

cf1 = axes[0].contourf(X, Y, Z_num, levels=30, cmap="viridis")
plt.colorbar(cf1, ax=axes[0])
axes[0].set_title("Solución numérica $V_{num}$")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

cf2 = axes[1].contourf(X, Y, Z_exa, levels=30, cmap="viridis")
plt.colorbar(cf2, ax=axes[1])
axes[1].set_title("Solución analítica $(x-y)^2$")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

cf3 = axes[2].contourf(X, Y, Z_err, levels=30, cmap="hot_r")
plt.colorbar(cf3, ax=axes[2])
axes[2].set_title(f"Error absoluto  (máx = {error.max():.2e})")
axes[2].set_xlabel("x"); axes[2].set_ylabel("y")

plt.tight_layout()
out_2d = os.path.join("imag", f"comparacion_{method_name}_2d.png")
plt.savefig(out_2d, dpi=150, bbox_inches="tight")
print(f"Guardada: {out_2d}")
plt.close(fig) # Liberar memoria

# -----------------------------------------------------------------------
# Figura 2: superficies 3D
# -----------------------------------------------------------------------
fig2 = plt.figure(figsize=(14, 5))
fig2.suptitle(f"Superficies 3D ({method_name}) — $\\nabla^2 V = 4$", fontsize=13)

ax1 = fig2.add_subplot(121, projection="3d")
ax1.plot_surface(X, Y, Z_num, cmap="viridis", alpha=0.9)
ax1.set_title("Solución numérica")
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("V")

ax2 = fig2.add_subplot(122, projection="3d")
ax2.plot_surface(X, Y, Z_exa, cmap="plasma", alpha=0.9)
ax2.set_title("Solución analítica $(x-y)^2$")
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("V")

plt.tight_layout()
out_3d = os.path.join("imag", f"superficies_{method_name}_3d.png")
plt.savefig(out_3d, dpi=150, bbox_inches="tight")
print(f"Guardada: {out_3d}")
plt.close(fig2) # Liberar memoria

print("-" * 50)
