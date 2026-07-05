import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error

RUTA_DATOS  = "./output/nba_features_target.csv"
RUTA_OUTPUT = "./output/experimentos"
os.makedirs(RUTA_OUTPUT, exist_ok=True)

SEED    = 42
K_FOLDS = 5
NOMBRE  = "Features_Temporales"

FEATURES_BASE = ["PER", "TS%", "USG%", "BPM", "VORP", "WS", "OWS", "DWS",
                  "PTS", "TRB", "AST", "STL", "BLK", "TOV", "Age", "G", "MP"]

print(f"EXPERIMENTO: {NOMBRE}")


df = pd.read_csv(RUTA_DATOS)
df = df.sort_values(["Player", "season"]).reset_index(drop=True)

df["PER_lag1"] = df.groupby("Player")["PER"].shift(1)
df["PER_lag2"] = df.groupby("Player")["PER"].shift(2)

df["PER_tendencia"] = df["PER"] - df["PER_lag1"]

df["PER_promedio_2y"] = df.groupby("Player")["PER"].transform(
    lambda x: x.rolling(window=2, min_periods=1).mean()
)

df["MP_lag1"] = df.groupby("Player")["MP"].shift(1)
df["MP_tendencia"] = df["MP"] - df["MP_lag1"]

df["temporadas_jugadas"] = df.groupby("Player").cumcount() + 1

FEATURES_TEMPORALES = ["PER_lag1", "PER_lag2", "PER_tendencia",
                        "PER_promedio_2y", "MP_lag1", "MP_tendencia",
                        "temporadas_jugadas"]
FEATURES_TOTAL = FEATURES_BASE + FEATURES_TEMPORALES

print(f"  Features temporales agregadas: {FEATURES_TEMPORALES}")
print(f"  Total features: {len(FEATURES_TOTAL)} "
      f"({len(FEATURES_BASE)} base + {len(FEATURES_TEMPORALES)} temporales)")

n_sin_lag1 = df["PER_lag1"].isna().sum()
print(f"  Filas sin historial previo (primera temporada del jugador en el "
      f"dataset): {n_sin_lag1} de {len(df)} ({n_sin_lag1/len(df)*100:.1f}%)\n")

features_disponibles = [f for f in FEATURES_TOTAL if f in df.columns]
X = df[features_disponibles].copy()
y = df["PER_next"].copy()

season_corte = df["season"].max() - 2
X_train = X[df["season"] <= season_corte]
X_test  = X[df["season"] >  season_corte]
y_train = y[df["season"] <= season_corte]
y_test  = y[df["season"] >  season_corte]

print(f"  Train: {len(X_train)} registros")
print(f"  Test:  {len(X_test)} registros\n")

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)


modelos = {
    "Regresion_Lineal": LinearRegression(),
    "Ridge":            Ridge(alpha=5.0),
    "Lasso":            Lasso(alpha=0.01, max_iter=3000, random_state=SEED),
    "MLP":              MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                      solver="adam", alpha=0.01, max_iter=3000,
                                      random_state=SEED, early_stopping=True,
                                      validation_fraction=0.15, n_iter_no_change=20),
}

resultados_con_temporal = {}

for nombre, modelo in modelos.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   modelo),
    ])
    cv_r2   = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    cv_rmse = np.sqrt(-cross_val_score(pipe, X_train, y_train,
                                        cv=kf, scoring="neg_mean_squared_error"))
    pipe.fit(X_train, y_train)
    y_pred    = pipe.predict(X_test)
    test_r2   = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    resultados_con_temporal[nombre] = {
        "cv_r2": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
        "cv_rmse": cv_rmse.mean(), "test_r2": test_r2, "test_rmse": test_rmse,
        "y_pred": y_pred, "pipe": pipe,
    }
    print(f"  {nombre:<20} → CV R²={cv_r2.mean():.4f}  Test R²={test_r2:.4f}  Test RMSE={test_rmse:.4f}")

RESULTADOS_SIN_TEMPORAL = {
    "Regresion_Lineal": {"test_r2": 0.7062, "test_rmse": 2.4148},
    "Ridge":             {"test_r2": 0.7063, "test_rmse": 2.4145},
    "Lasso":             {"test_r2": 0.7047, "test_rmse": 2.4211},
    "MLP":               {"test_r2": 0.7047, "test_rmse": 2.4210},
}

print("COMPARACIÓN: SIN features temporales vs CON features temporales")
print(f"{'Modelo':<20} {'R² sin temp.':>13} {'R² con temp.':>13} {'Mejora':>9}")

mejoras = {}
for nombre in modelos.keys():
    r2_sin = RESULTADOS_SIN_TEMPORAL[nombre]["test_r2"]
    r2_con = resultados_con_temporal[nombre]["test_r2"]
    mejora = r2_con - r2_sin
    mejoras[nombre] = mejora
    print(f"{nombre:<20} {r2_sin:>13.4f} {r2_con:>13.4f} {mejora:>+9.4f}")

mejor_modelo = max(resultados_con_temporal, key=lambda n: resultados_con_temporal[n]["test_r2"])
print(f"\n Mejor modelo con features temporales: {mejor_modelo} "
      f"(Test R²={resultados_con_temporal[mejor_modelo]['test_r2']:.4f})")


print("Relevancia de features temporales según Lasso")

lasso_pipe = resultados_con_temporal["Lasso"]["pipe"]
lasso_model = lasso_pipe.named_steps["model"]
coefs = pd.Series(lasso_model.coef_, index=features_disponibles)
activos = coefs[coefs != 0].sort_values(key=abs, ascending=False)

print("Coeficientes de features TEMPORALES:")
for feat in FEATURES_TEMPORALES:
    if feat in coefs.index:
        val = coefs[feat]
        estado = "ACTIVA" if val != 0 else "eliminada (coef=0)"
        print(f"  {feat:<20} coef={val:+.4f}  [{estado}]")


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Features Temporales: ¿Ayuda la trayectoria del jugador? — Predicción de PER NBA\n"
             "Introducción al Aprendizaje Automático · UTFSM 2026",
             fontsize=12, fontweight="bold")

COLORES = {"Regresion_Lineal": "#378ADD", "Ridge": "#1D9E75",
           "Lasso": "#D85A30", "MLP": "#7F77DD"}
NOMBRES_CORTOS = {"Regresion_Lineal": "Reg. Lineal", "Ridge": "Ridge",
                   "Lasso": "Lasso", "MLP": "MLP"}

ax1 = axes[0]
x_pos = np.arange(len(modelos))
width = 0.35
r2_sin_lista = [RESULTADOS_SIN_TEMPORAL[n]["test_r2"] for n in modelos]
r2_con_lista = [resultados_con_temporal[n]["test_r2"] for n in modelos]
ax1.bar(x_pos - width/2, r2_sin_lista, width, label="Sin features temporales",
        color="#BBBBBB", alpha=0.85, edgecolor="white")
ax1.bar(x_pos + width/2, r2_con_lista, width, label="Con features temporales",
        color=[COLORES[n] for n in modelos], alpha=0.9, edgecolor="white")
ax1.axhline(0.80, color="red", linestyle="--", linewidth=1.2, label="Objetivo R²=0.80")
ax1.set_xticks(x_pos)
ax1.set_xticklabels([NOMBRES_CORTOS[n] for n in modelos])
ax1.set_ylabel("Test R²")
ax1.set_title("Comparación: Sin vs Con Features Temporales")
ax1.legend(fontsize=8)
ax1.set_ylim(0, 0.9)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.spines[["top","right"]].set_visible(False)
for i, (sin, con) in enumerate(zip(r2_sin_lista, r2_con_lista)):
    ax1.text(i - width/2, sin + 0.01, f"{sin:.3f}", ha="center", fontsize=7.5)
    ax1.text(i + width/2, con + 0.01, f"{con:.3f}", ha="center", fontsize=7.5, fontweight="bold")

ax2 = axes[1]
y_pred_mejor = resultados_con_temporal[mejor_modelo]["y_pred"]
ax2.scatter(y_test, y_pred_mejor, alpha=0.45, color=COLORES[mejor_modelo], s=20, edgecolors="none")
lims = [min(y_test.min(), y_pred_mejor.min())-1, max(y_test.max(), y_pred_mejor.max())+1]
ax2.plot(lims, lims, "k--", linewidth=1, label="Predicción perfecta")
ax2.set_xlabel("PER real (t+1)")
ax2.set_ylabel("PER predicho")
ax2.set_title(f"Real vs Predicho — {NOMBRES_CORTOS[mejor_modelo]} (con temporales)")
ax2.legend(fontsize=9)
ax2.grid(linestyle="--", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)
ax2.text(0.05, 0.92, f"R²={resultados_con_temporal[mejor_modelo]['test_r2']:.3f}\n"
                      f"RMSE={resultados_con_temporal[mejor_modelo]['test_rmse']:.3f}",
         transform=ax2.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

ax3 = axes[2]
coefs_temporales = coefs[FEATURES_TEMPORALES].sort_values(key=abs)
colores_barra = ["#1D9E75" if v > 0 else "#D85A30" for v in coefs_temporales.values]
ax3.barh(coefs_temporales.index, coefs_temporales.values, color=colores_barra,
         alpha=0.85, edgecolor="white")
ax3.axvline(0, color="black", linewidth=0.8)
ax3.set_xlabel("Coeficiente (Lasso)")
ax3.set_title("Relevancia de Features Temporales (Lasso)")
ax3.grid(axis="x", linestyle="--", alpha=0.3)
ax3.spines[["top","right"]].set_visible(False)
ax3.tick_params(axis="y", labelsize=8.5)

plt.tight_layout()
ruta_fig = os.path.join(RUTA_OUTPUT, f"{NOMBRE}_resultados.png")
plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")

tabla = pd.DataFrame([
    {
        "Modelo": n,
        "Test R² (sin temporales)": RESULTADOS_SIN_TEMPORAL[n]["test_r2"],
        "Test R² (con temporales)": round(resultados_con_temporal[n]["test_r2"], 4),
        "Mejora R²": round(mejoras[n], 4),
        "Test RMSE (con temporales)": round(resultados_con_temporal[n]["test_rmse"], 4),
        "CV R² (con temporales)": round(resultados_con_temporal[n]["cv_r2"], 4),
    }
    for n in modelos
])
tabla.to_csv(os.path.join(RUTA_OUTPUT, f"{NOMBRE}_metricas.csv"), index=False)


print("CONCLUSIÓN DEL EXPERIMENTO")

mejora_promedio = np.mean(list(mejoras.values()))
print(f" Mejora promedio en Test R² across los 4 modelos: {mejora_promedio:+.4f}")
if mejora_promedio > 0.02:
    print(" Las features temporales aportan una mejora considerable.")
elif mejora_promedio > 0:
    print(" Las features temporales aportan una mejora marginal.")
else:
    print(" Las features temporales NO mejoran el desempeño; refuerza la")
    print(" hipótesis de un techo informacional independiente del contexto histórico.")