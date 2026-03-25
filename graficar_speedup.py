import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------------------
# Datos obtenidos de las ejecuciones (Tiempo base secuencial: 1657.98 s)
# -----------------------------------------------------------------------
t_serial = 1657.98

# Diccionario con TODOS los tiempos actualizados
tiempos = {
    "parallel_for": 102.39,
    "collapse(2)": 138.58,
    "sections": 103.04,
    "static": 110.46,
    "dynamic": 129.43,
    "sync": 115.02,
    "sync_sp": 98.66,    # ¡El nuevo ganador!
    "task": 108.64,
    "critical": 427.79,  # El más lento por el cuello de botella
    "atomic": 124.60
}

# Calcular el Speedup (T_serial / T_paralelo)
nombres = list(tiempos.keys())
valores_speedup = [t_serial / t for t in tiempos.values()]

# -----------------------------------------------------------------------
# Configuración de la gráfica
# -----------------------------------------------------------------------
plt.figure(figsize=(12, 6))

# Crear barras con un mapa de color
colores = plt.cm.viridis(np.linspace(0.9, 0.1, len(nombres)))
barras = plt.bar(nombres, valores_speedup, color=colores, edgecolor='black', zorder=3)

# Añadir el valor numérico encima de cada barra
for barra in barras:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2., altura + 0.3,
             f'{altura:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Línea ideal (Si 20 hilos dieran 20x de aceleración perfecta)
plt.axhline(y=20, color='red', linestyle='--', linewidth=1.5, zorder=2, label="Speedup Ideal (20x)")

# Detalles estéticos
plt.title('Aceleración (Speedup) de Directivas OpenMP vs Secuencial\n(Malla 1024x1024, 20 Hilos)', fontsize=14, pad=15)
plt.ylabel('Speedup ($T_{sec} / T_{par}$)', fontsize=12)
plt.xlabel('Estrategia de Paralelización', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
plt.legend()
plt.ylim(0, 22) # Dar espacio para los textos y la línea de 20x

plt.tight_layout()

# Guardar en la carpeta imag/
os.makedirs("imag", exist_ok=True)
ruta_salida = os.path.join("imag", "speedup_comparativo.png")
plt.savefig(ruta_salida, dpi=150)
print(f"Gráfica de Speedup guardada exitosamente en: {ruta_salida}")
