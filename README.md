# nba-per-prediction
# Predicción del Rendimiento de Jugadores NBA mediante Aprendizaje Supervisado

**INF398 — Introducción al Aprendizaje Automático · UTFSM, Semestre I 2026**
**Autor:** Jose Miguel Guerrero Cabrera · Rol 202004119-9

---

## Descripción del proyecto

Este proyecto predice el **Player Efficiency Rating (PER)** de jugadores de la NBA en la temporada `t+1` a partir de sus estadísticas de la temporada `t`. Se evalúan y comparan cuatro modelos de regresión supervisada —Regresión Lineal, Ridge, Lasso y MLP a través de cinco bloques experimentales que incluyen interacciones polinomiales, optimización exhaustiva de hiperparámetros y features de contexto temporal.

**Mejor resultado obtenido:** Ridge Regression con features temporales → Test R²=0.7159, RMSE=2.3744

---

## Estructura del repositorio

```
nba-per-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── README.md               # instrucciones para obtener los datos
├── src/
│   ├── paso1_limpieza_datos.py # pipeline de preprocesamiento
│   ├── exp1_regresion_lineal.py
│   ├── exp2_ridge.py
│   ├── exp3_lasso.py
│   ├── exp4_mlp.py
│   ├── exp4_mlp_v2_mejorado.py # MLP con lbfgs + ensemble
│   ├── exp6_polinomial.py
│   ├── exp7_mlp_gridsearch.py
│   └── exp9_features_temporales.py
├── output/
│   └── experimentos/           # CSVs y gráficos generados
└── report/
    ├── main.tex
    └── figuras/
```}

---

## Requisitos de software

- Python 3.9 o superior
- pip

Instalar todas las dependencias con:

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
scikit-learn>=1.2
```


---

## Obtención de los datos

Los datasets **no se incluyen en el repositorio** por su tamaño. Deben descargarse manualmente desde las siguientes fuentes:

### Basketball-Reference.com (fuente principal)

Descargar **Advanced Stats** y **Per Game Stats** para las temporadas 2014-15 a 2024-25:

1. Ir a: `https://www.basketball-reference.com/leagues/NBA_YYYY_advanced.html`
2. Hacer clic en **Share & Export → Get table as CSV**
3. Guardar cada archivo como `Nba_advanced_YY.csv`
4. Repetir para Per Game Stats:
   `https://www.basketball-reference.com/leagues/NBA_YYYY_per_game.html`
   Guardar como `NBA_PerGame_YY.csv`

En total se necesitan **22 archivos CSV** (11 temporadas × 2 tablas). Colocarlos en la carpeta `league/`.

### Kaggle — NBA Players Stats (respaldo histórico)

- URL: https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats
- Descargar el dataset completo y extraerlo en `league/Kaggle/`
- Se usan principalmente: `Advanced.csv` y `Player Per Game.csv`

---

## Instrucciones para ejecutar el proyecto

Todos los scripts se ejecutan desde la carpeta raíz del repositorio.

### Paso 1 — Limpieza y construcción del dataset

```bash
python src/Data.py
```

Genera en `output/`:
- `nba_dataset_ml.csv` — dataset completo
- `nba_features_target.csv` — 17 features + columna `PER_next` (variable objetivo)

**Este paso es obligatorio antes de cualquier experimento.**

---

### Paso 2 — Experimentos base (4 modelos)

```bash
python src/Regresion_lineal.py   # Regresión Lineal (baseline)
python src/Ridge.py              # Ridge Regression
python src/Lasso.py              # Lasso Regression
python src/MLP.py                # MLP feedforward (adam)
```

Cada script genera en `output/experimentos/`:
- `<Modelo>_resultados.png` — gráficos (real vs predicho, curvas de aprendizaje)
- `<Modelo>_metricas.csv` — métricas (CV R², Test R², RMSE)

---

### Paso 3 — Experimentos adicionales

```bash
# Interacciones polinomiales (Ridge y Lasso con 153 features)
python src/Polinomial.py

# GridSearchCV exhaustivo para MLP (32 combinaciones × 5 folds, ~10-15 min)
python src/MLP_Gridsearch.py

# Features temporales: lag, tendencia y promedio móvil (MEJOR RESULTADO)
python src/MLP_Temporal.py
```

---

## Resultados principales

| Modelo | Test R² | Test RMSE |
|--------|---------|-----------|
| **Ridge + features temporales** | **0.7159** | **2.3744** |
| Regresión Lineal + features temporales | 0.7158 | 2.3751 |
| Lasso + features temporales | 0.7155 | 2.3762 |
| MLP lbfgs + ensemble | 0.7140 | — |
| Lasso Polinomial | 0.7081 | 2.4071 |
| Ridge (base) | 0.7063 | 2.4145 |
| Regresión Lineal (base) | 0.7062 | 2.4148 |

El techo de R²≈0.71 se mantiene robusto a través de los 5 bloques experimentales, apuntando a un límite informacional en las estadísticas disponibles más que a una limitación algorítmica.

---

## Referencia

El informe completo en PDF se encuentra en `report/informe_final.pdf`.
