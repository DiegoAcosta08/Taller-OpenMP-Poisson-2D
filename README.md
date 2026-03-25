# Taller OpenMP — Solución de la Ecuación de Poisson 2D

Implementación y comparativa de rendimiento de diferentes estrategias de paralelización con **OpenMP** para resolver la ecuación de Poisson en 2D mediante el método iterativo de Jacobi, sobre un enmallado de **1024 × 1024** nodos.

---

## Descripción del problema

La ecuación de Poisson 2D se resuelve iterativamente usando el esquema de diferencias finitas de Jacobi:

```
u[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - h² * f[i][j])
```

El criterio de convergencia se basa en el delta máximo entre iteraciones sucesivas. Se utiliza un dominio cuadrado con condiciones de frontera de Dirichlet y un término fuente `f(x,y)`.

---

## Estructura del repositorio

```
.
├── Makefile
├── README.md
├── src/
│   ├── poisson_serial.c                  # Actividad 1 — Versión secuencial
│   ├── poisson_parallel_for.c       # Actividad 2 — Paralelo básico
│   ├── poisson_collapse.c             # Actividad 3 — Colapsado de bucles
│   ├── poisson_sections.c        # Actividad 4 — Inicialización y fuente en paralelo
│   ├── poisson_static.c      # Actividad 5a — Schedule static
│   ├── poisson_dynamic.c     # Actividad 5b — Schedule dynamic
│   ├── poisson_sync_sp.c         # Actividad 5c — Control de sincronización (sin print)
│   ├── poisson_sync.c           # Actividad 5d — Control de sincronización (con print)
│   ├── poisson_critical.c             # Actividad 6a — omp critical
│   ├── poisson_atomic.c               # Actividad 6b — omp atomic
│   └── poisson_task.c                # Actividad 7 — Tasks
└── results/
```

---

## Compilación

```bash
# Compilar todas las versiones
make all

# Compilar una versión específica, por ejemplo:
make poisson_serial
make poisson_parallel_for

# Limpiar binarios
make clean
```

El `Makefile` usa `g++` con las flags `-fopenmp -O3 -lm`.

---

## Ejecución

```bash
# Ejemplo de ejecución (20 hilos por defecto en versiones paralelas)
./poisson_serial
./poisson_parallel_for
```

El número de hilos puede controlarse con la variable de entorno `OMP_NUM_THREADS`:

```bash
export OMP_NUM_THREADS=20
./poisson_parallel_for
```

---

## Resultados — Enmallado 1024 × 1024

Todas las versiones paralelas se ejecutaron con **20 hilos**. Los tiempos reportados corresponden al promedio de ejecución sobre la convergencia completa del método de Jacobi.

| # | Versión | Directivas OpenMP utilizadas | Tiempo (s) | Iteraciones | Hilos |
|---|---------|------------------------------|:----------:|:-----------:|:-----:|
| 1 | **Secuencial** | N/A | 1657.98 | 236161 | 1 |
| 2 | **Paralelo básico** | `parallel for` + `reduction(max:delta)` | 102.39 | 236164 | 20 |
| 3 | **Colapsado de bucles** | `parallel for collapse(2)` + `reduction(max:delta)` | 138.58 | 236161 | 20 |
| 4 | **Inicialización y fuente en paralelo** | `parallel for` + `reduction` + `section` | 103.04 | 236161 | 20 |
| 5a | **Control explícito + schedule static** | `parallel shared` + `for reduction` + `schedule(static)` | 110.46 | 236163 | 20 |
| 5b | **Control explícito + schedule dynamic** | `parallel shared` + `for reduction` + `schedule(dynamic)` | 129.43 | 275238 | 20 |
| 5c | **Control de sincronización (sin print)** | `parallel shared` + `single` + `for reduction` + `schedule(static)` + `nowait` + `critical` | 115.02 | 236172 | 20 |
| 5d | **Control de sincronización (con print)** | `parallel shared` + `single` + `for reduction` + `schedule(static)` + `nowait` + `critical` | 501.87 | 236501 | 20 |
| 6a | **omp critical** | `for reduction` + `omp critical` | 105.86 | 236166 | 20 |
| 6b | **omp atomic** | `for reduction` + `omp atomic` | 105.87 | 236164 | 20 |
| 6c | **Tasks** | `omp task` | 114.21 | 244480 | 20 |

### Speedup respecto a la versión secuencial

| Versión | Tiempo (s) | Speedup |
|---------|:----------:|:-------:|
| Paralelo básico | 102.39 | **~16.2×** |
| Inicialización y fuente en paralelo | 103.04 | ~16.1× |
| Schedule static | 110.46 | ~15.0× |
| omp critical | 105.86 | ~15.7× |
| omp atomic | 105.87 | ~15.7× |
| Tasks | 114.21 | ~14.5× |
| Control de sincronización (sin print) | 115.02 | ~14.4× |
| Schedule dynamic | 129.43 | ~12.8× |
| Colapsado de bucles | 138.58 | ~12.0× |
| **Control de sincronización (con print)** | **501.87** | **~3.3×** |

---

## Análisis de resultados

- La **versión paralela básica** obtiene el mejor balance entre simplicidad y rendimiento (~16.2× de speedup), siendo la referencia más competitiva del taller.

- La **inicialización y fuente en paralelo** alcanza un speedup muy similar al básico (~16.1×), confirmando que paralelizar también la fase de inicialización no introduce overhead significativo y puede ser una buena práctica en problemas de mayor escala.

- El **colapsado de bucles** (`collapse(2)`) resultó más lento de lo esperado (~12.0× vs ~16.2×), posiblemente por el overhead de repartición del espacio de iteración combinado y un peor aprovechamiento de la localidad de caché respecto al paralelo básico.

- El **schedule dynamic** incrementa las iteraciones hasta la convergencia (275 238 vs ~236 161 del resto), lo que indica un comportamiento no determinista en el orden de actualización de las celdas, además de ser más lento (~12.8×) por el costo de asignación dinámica de chunks en cada iteración.

- La comparación entre las versiones **sin print** (115.02 s) y **con print** (501.87 s) es el resultado más llamativo del taller: incluir un `printf` dentro de la región paralela degrada el rendimiento de forma drástica, elevando el tiempo de ejecución en más de 4 veces respecto a la versión sin print, y reduciendo el speedup de ~14.4× a apenas ~3.3×. Esto se debe a que `printf` no es thread-safe y la directiva `critical` que lo protege serializa completamente esa sección en cada una de las 236 501 iteraciones, convirtiendo la E/S en un cuello de botella dominante que prácticamente anula el beneficio del paralelismo. Significativamente, esta versión termina siendo incluso más lenta que muchas otras versiones paralelas, y se acerca peligrosamente al rendimiento secuencial. Es un ejemplo contundente de por qué debe evitarse la E/S bloqueante protegida con `critical` en lazos de cómputo intensivo.

- Las directivas `omp critical` y `omp atomic` muestran un rendimiento prácticamente idéntico (105.86 s vs 105.87 s). Para operaciones de reducción simples como el cálculo del delta máximo, `atomic` es la opción preferible por su menor granularidad y overhead de sincronización, aunque en este caso el compilador optimiza ambas de forma equivalente.

- El uso de **tasks** añade overhead de creación y gestión de tareas sin ventaja significativa para este patrón de acceso regular y estructurado, resultando algo más lento que el paralelo básico (~14.5× vs ~16.2×).

---

## Requisitos

- GCC ≥ 9 con soporte OpenMP (`-fopenmp`)
- Sistema Linux (probado en Ubuntu 22.04)
- Al menos 4 núcleos físicos recomendados para explotar el paralelismo

---

## Dependencias

```bash
sudo apt-get install gcc libomp-dev
```

---

## Autores

Diego Acosta y Julián Cogua

Taller desarrollado como parte del curso de **Computación de Alto Rendimiento** — Programación paralela con OpenMP.
