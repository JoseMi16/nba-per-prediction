import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error

RUTA_DATOS  = "./output/nba_features_target.csv"
RUTA_OUTPUT = "./output/experimentos"
os.makedirs(RUTA_OUTPUT, exist_ok=True)

SEED    = 42
K_FOLDS = 5
NOMBRE  = "Regresion_Lineal"
COLOR   = "#378ADD"

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
print(f"  Test:  {len(X_test)} registros")
print(f"  Features: {len(features_disponibles)}\n")

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   LinearRegression())
])

print("Validación cruzada")
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

cv_r2   = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2")
cv_rmse = np.sqrt(-cross_val_score(pipeline, X_train, y_train,
                                   cv=kf, scoring="neg_mean_squared_error"))

print(f"  CV R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"  CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}\n")

pipeline.fit(X_train, y_train)
y_pred    = pipeline.predict(X_test)
test_r2   = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Evaluación en test:")
print(f"  Test R²:   {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}\n")

modelo = pipeline.named_steps["model"]
coefs  = pd.Series(np.abs(modelo.coef_), index=features_disponibles).sort_values(ascending=False)

print("Top 10 coeficientes:")
for feat, val in coefs.head(10).items():
    print(f"  {feat:<12} {val:.4f}")

train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train,
    cv=kf, scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"{NOMBRE} — Predicción de PER NBA\nIntroducción al Aprendizaje Automático · UTFSM 2026",
             fontsize=12, fontweight="bold")

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
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)
ax2.plot(train_sizes, train_mean, "o-", color=COLOR, label="Train R²")
ax2.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color=COLOR)
ax2.plot(train_sizes, val_mean, "s--", color="#D85A30", label="Validación R²")
ax2.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color="#D85A30")
ax2.set_xlabel("Tamaño del conjunto de entrenamiento")
ax2.set_ylabel("R²")
ax2.set_title("Curva de Aprendizaje")
ax2.legend(fontsize=9)
ax2.grid(linestyle="--", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

ax3 = axes[2]
top10 = coefs.head(10)
bars = ax3.barh(top10.index[::-1], top10.values[::-1], color=COLOR, alpha=0.8, edgecolor="white")
ax3.set_xlabel("|Coeficiente|")
ax3.set_title("Importancia de Features\n(|coeficiente| escalado)")
ax3.grid(axis="x", linestyle="--", alpha=0.3)
ax3.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars, top10.values[::-1]):
    ax3.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout()
ruta_fig = os.path.join(RUTA_OUTPUT, f"{NOMBRE}_resultados.png")
plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")

resultado = pd.DataFrame([{
    "Modelo":        NOMBRE,
    "CV R² media":   round(cv_r2.mean(), 4),
    "CV R² std":     round(cv_r2.std(), 4),
    "CV RMSE media": round(cv_rmse.mean(), 4),
    "CV RMSE std":   round(cv_rmse.std(), 4),
    "Test R²":       round(test_r2, 4),
    "Test RMSE":     round(test_rmse, 4),
}])
resultado.to_csv(os.path.join(RUTA_OUTPUT, f"{NOMBRE}_metricas.csv"), index=False)