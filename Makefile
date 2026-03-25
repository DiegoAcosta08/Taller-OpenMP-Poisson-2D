# Compilador y banderas
CXX = g++
CXXFLAGS = -O3 -fopenmp -Wall
THREADS = 20

# Directorios
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data
IMG_DIR = imag

# Archivos fuente y ejecutables
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
EXECUTABLES = $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%, $(SOURCES))

# Regla por defecto: hace todo el proceso de inicio a fin
all: dirs compile run plot

# 1. Crear directorios necesarios
dirs:
	mkdir -p $(BIN_DIR) $(DATA_DIR) $(IMG_DIR)

# 2. Compilar todos los archivos .cpp y guardarlos en bin/
compile: $(EXECUTABLES)

$(BIN_DIR)/%: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# 3. Ejecutar todos los binarios y mover los .dat generados a data/
run:
	@for exe in $(EXECUTABLES); do \
		echo "========================================"; \
		echo "Ejecutando $$exe con $(THREADS) hilos..."; \
		OMP_NUM_THREADS=$(THREADS) ./$$exe; \
	done
	@echo "Moviendo archivos de datos a $(DATA_DIR)/..."
	@mv *.dat $(DATA_DIR)/ 2>/dev/null || true

# 4. Generar las gráficas iterando sobre los archivos .dat
plot:
	@echo "========================================"
	@echo "Generando gráficas..."
	@for dat in $(DATA_DIR)/*.dat; do \
		python3 graficar.py $$dat; \
	done

# Limpiar archivos generados (opcional, para reiniciar)
clean:
	rm -rf $(BIN_DIR) $(DATA_DIR) $(IMG_DIR)
