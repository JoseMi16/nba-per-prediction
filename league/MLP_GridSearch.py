import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error

RUTA_DATOS  = "./output/nba_features_target.csv"
RUTA_OUTPUT = "./output/experimentos"
os.makedirs(RUTA_OUTPUT, exist_ok=True)

SEED    = 42
K_FOLDS = 5
NOMBRE  = "MLP_GridSearch"
COLOR   = "#7F77DD"

FEATURES = ["PER", "TS%", "USG%", "BPM", "VORP", "WS", "OWS", "DWS",
            "PTS", "TRB", "AST", "STL", "BLK", "TOV", "Age", "G", "MP"]

print(f"EXPERIMENTO: {NOMBRE}")

df = pd.read_csv(RUTA_DATOS)
features_disponibles = [f for f in FEATURES if f in df.columns]

X = df[features_disponibles].copy()
y = df["PER_next"].copy()

season_corte = df["season"].max() - 2
X_train = X[df["season"] <= season_corte]
X_test  = X[df["season"] >  season_corte]
y_train = y[df["season"] <= season_corte]
y_test  = y[df["season"] >  season_corte]

print(f"  Train: {len(X_train)} registros")
print(f"  Test:  {len(X_test)} registros\n")

param_grid = {
    "model__hidden_layer_sizes": [(32,), (64, 32), (128, 64), (64, 32, 16)],
    "model__alpha":              [0.0001, 0.001, 0.01, 0.1],
    "model__learning_rate_init": [0.001, 0.01],
}

n_combinaciones = (len(param_grid["model__hidden_layer_sizes"]) *
                    len(param_grid["model__alpha"]) *
                    len(param_grid["model__learning_rate_init"]))
print(f"  Combinaciones a evaluar: {n_combinaciones} × {K_FOLDS} folds = "
      f"{n_combinaciones * K_FOLDS} entrenamientos\n")

pipeline_base = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   MLPRegressor(
        activation="relu",
        solver="adam",
        max_iter=800,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    ))
])

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)


grid_search = GridSearchCV(
    estimator=pipeline_base,
    param_grid=param_grid,
    cv=kf,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print(f"\n Mejores hiperparámetros encontrados:")
for k, v in grid_search.best_params_.items():
    print(f"     {k}: {v}")
print(f" Mejor CV R²: {grid_search.best_score_:.4f}\n")

resultados_grid = pd.DataFrame(grid_search.cv_results_)
top5 = resultados_grid.sort_values("mean_test_score", ascending=False).head(5)

print("Mejores combinaciones de hiperparámetros:")
for _, row in top5.iterrows():
    print(f"  {row['params']} → CV R²={row['mean_test_score']:.4f} "
          f"(±{row['std_test_score']:.4f})")

mejor_modelo = grid_search.best_estimator_

y_pred    = mejor_modelo.predict(X_test)
test_r2   = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nEvaluación en test:")
print(f"  Test R²:   {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}\n")

train_sizes, train_scores, val_scores = learning_curve(
    mejor_modelo, X_train, y_train,
    cv=kf, scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"MLP Optimizado {grid_search.best_params_['model__hidden_layer_sizes']} "
             f"— Predicción de PER NBA\nIntroducción al Aprendizaje Automático · UTFSM 2026",
             fontsize=11, fontweight="bold")

ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.45, color=COLOR, s=20, edgecolors="none")
lims = [min(y_test.min(), y_pred.min())-1, max(y_test.max(), y_pred.max())+1]
ax1.plot(lims, lims, "k--", linewidth=1, label="Predicción perfecta")
ax1.set_xlabel("PER real (t+1)")
ax1.set_ylabel("PER predicho")
ax1.set_title("Real vs Predicho")
ax1.legend(fontsize=9)
ax1.grid(linestyle="--", alpha=0.3)
ax1.spines[["top","right"]].set_visible(False)
ax1.text(0.05, 0.92, f"R²={test_r2:.3f}\nRMSE={test_rmse:.3f}",
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

ax2 = axes[1]
pivot = resultados_grid.pivot_table(
    values="mean_test_score",
    index="param_model__alpha",
    columns="param_model__hidden_layer_sizes",
    aggfunc="mean"
)
im = ax2.imshow(pivot.values, cmap="viridis", aspect="auto")
ax2.set_xticks(range(len(pivot.columns)))
ax2.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right", fontsize=8)
ax2.set_yticks(range(len(pivot.index)))
ax2.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
ax2.set_xlabel("Arquitectura")
ax2.set_ylabel("Alpha (regularización)")
ax2.set_title("CV R² promedio por combinación")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax2.text(j, i, f"{val:.3f}", ha="center", va="center",
                     color="white" if val < pivot.values.mean() else "black", fontsize=7)
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

ax3 = axes[2]
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)
ax3.plot(train_sizes, train_mean, "o-", color=COLOR, label="Train R²")
ax3.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color=COLOR)
ax3.plot(train_sizes, val_mean, "s--", color="#D85A30", label="Validación R²")
ax3.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color="#D85A30")
ax3.set_xlabel("Tamaño del conjunto de entrenamiento")
ax3.set_ylabel("R²")
ax3.set_title("Curva de Aprendizaje (mejor modelo)")
ax3.legend(fontsize=9)
ax3.grid(linestyle="--", alpha=0.3)
ax3.spines[["top","right"]].set_visible(False)

plt.tight_layout()
ruta_fig = os.path.join(RUTA_OUTPUT, f"{NOMBRE}_resultados.png")
plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")

resultado = pd.DataFrame([{
    "Modelo":               NOMBRE,
    "Mejor arquitectura":   str(grid_search.best_params_["model__hidden_layer_sizes"]),
    "Mejor alpha":          grid_search.best_params_["model__alpha"],
    "Mejor learning_rate":  grid_search.best_params_["model__learning_rate_init"],
    "CV R² (GridSearch)":   round(grid_search.best_score_, 4),
    "Test R²":              round(test_r2, 4),
    "Test RMSE":            round(test_rmse, 4),
}])
resultado.to_csv(os.path.join(RUTA_OUTPUT, f"{NOMBRE}_metricas.csv"), index=False)


resultados_grid[[
    "param_model__hidden_layer_sizes", "param_model__alpha",
    "param_model__learning_rate_init", "mean_test_score", "std_test_score"
]].sort_values("mean_test_score", ascending=False).to_csv(
    os.path.join(RUTA_OUTPUT, f"{NOMBRE}_grid_completo.csv"), index=False
)