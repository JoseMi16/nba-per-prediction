import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Lasso
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
NOMBRE  = "Lasso_Regression"
COLOR   = "#D85A30"

ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

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


kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
alpha_scores = {}

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
X_train_imp = imputer.fit_transform(X_train)
X_train_scl = scaler.fit_transform(X_train_imp)

for alpha in ALPHAS:
    modelo = Lasso(alpha=alpha, max_iter=5000, random_state=SEED)
    scores = cross_val_score(modelo, X_train_scl, y_train, cv=kf, scoring="r2")
    alpha_scores[alpha] = scores.mean()
    print(f"  alpha={alpha:<8} → CV R²={scores.mean():.4f}")

mejor_alpha = max(alpha_scores, key=alpha_scores.get)
print(f"\n Mejor alpha: {mejor_alpha} (CV R²={alpha_scores[mejor_alpha]:.4f})\n")

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   Lasso(alpha=mejor_alpha, max_iter=5000, random_state=SEED))
])

cv_r2   = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2")
cv_rmse = np.sqrt(-cross_val_score(pipeline, X_train, y_train,
                                   cv=kf, scoring="neg_mean_squared_error"))

print("Validación cruzada (mejor alpha):")
print(f"  CV R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"  CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}\n")

pipeline.fit(X_train, y_train)
y_pred    = pipeline.predict(X_test)
test_r2   = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Evaluación en test:")
print(f"  Test R²:   {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}\n")

modelo  = pipeline.named_steps["model"]
coefs   = pd.Series(modelo.coef_, index=features_disponibles)
activos = coefs[coefs != 0].sort_values(key=abs, ascending=False)
nulos   = coefs[coefs == 0]

print(f"Features seleccionadas por Lasso (coef ≠ 0): {len(activos)}/{len(features_disponibles)}")
for feat, val in activos.items():
    signo = "+" if val > 0 else "-"
    print(f"  {signo} {feat:<12} {val:.4f}")

print(f"\nFeatures eliminadas por Lasso (coef = 0): {len(nulos)}")
print(f"  {list(nulos.index)}")

train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train,
    cv=kf, scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"{NOMBRE} (α={mejor_alpha}) — Predicción de PER NBA\nIntroducción al Aprendizaje Automático · UTFSM 2026",
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
colores_coef = [COLOR if v > 0 else "#7F77DD" for v in activos.values]
bars = ax2.barh(activos.index[::-1], activos.values[::-1],
                color=colores_coef[::-1], alpha=0.85, edgecolor="white")
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Coeficiente")
ax2.set_title(f"Coeficientes Lasso\n({len(activos)} activos, {len(nulos)} eliminados)")
ax2.grid(axis="x", linestyle="--", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)

ax3 = axes[2]
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)
ax3.plot(train_sizes, train_mean, "o-", color=COLOR, label="Train R²")
ax3.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color=COLOR)
ax3.plot(train_sizes, val_mean, "s--", color="#378ADD", label="Validación R²")
ax3.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color="#378ADD")
ax3.set_xlabel("Tamaño del conjunto de entrenamiento")
ax3.set_ylabel("R²")
ax3.set_title("Curva de Aprendizaje")
ax3.legend(fontsize=9)
ax3.grid(linestyle="--", alpha=0.3)
ax3.spines[["top","right"]].set_visible(False)

plt.tight_layout()
ruta_fig = os.path.join(RUTA_OUTPUT, f"{NOMBRE}_resultados.png")
plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")

resultado = pd.DataFrame([{
    "Modelo":             NOMBRE,
    "Mejor alpha":        mejor_alpha,
    "Features activas":   len(activos),
    "Features eliminadas":len(nulos),
    "CV R² media":        round(cv_r2.mean(), 4),
    "CV R² std":          round(cv_r2.std(), 4),
    "CV RMSE media":      round(cv_rmse.mean(), 4),
    "CV RMSE std":        round(cv_rmse.std(), 4),
    "Test R²":            round(test_r2, 4),
    "Test RMSE":          round(test_rmse, 4),
}])
resultado.to_csv(os.path.join(RUTA_OUTPUT, f"{NOMBRE}_metricas.csv"), index=False)