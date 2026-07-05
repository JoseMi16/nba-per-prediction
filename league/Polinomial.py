import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

RUTA_DATOS  = "./output/nba_features_target.csv"
RUTA_OUTPUT = "./output/experimentos"
os.makedirs(RUTA_OUTPUT, exist_ok=True)

SEED    = 42
K_FOLDS = 5
NOMBRE  = "Ridge_Lasso_Polinomial"
COLOR_RIDGE = "#1D9E75"
COLOR_LASSO = "#D85A30"

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
print(f"  Features originales: {len(features_disponibles)}\n")

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

def construir_pipeline(modelo):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler1", StandardScaler()),
        ("poly",    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("scaler2", StandardScaler()),
        ("model",   modelo)
    ])


poly_check = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
n_features_poly = poly_check.fit_transform(X_train.fillna(0)).shape[1]
print(f"  Features tras expansión polinomial: {n_features_poly} "
      f"({len(features_disponibles)} originales + interacciones)\n")

print("RIDGE + Features Polinomiales")


ALPHAS_RIDGE = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0]
ridge_scores = {}

for alpha in ALPHAS_RIDGE:
    pipe = construir_pipeline(Ridge(alpha=alpha))
    scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    ridge_scores[alpha] = scores.mean()
    print(f"  alpha={alpha:<8} -> CV R²={scores.mean():.4f}")

mejor_alpha_ridge = max(ridge_scores, key=ridge_scores.get)
print(f"\n Mejor alpha Ridge: {mejor_alpha_ridge} (CV R²={ridge_scores[mejor_alpha_ridge]:.4f})\n")

pipe_ridge = construir_pipeline(Ridge(alpha=mejor_alpha_ridge))
cv_r2_ridge   = cross_val_score(pipe_ridge, X_train, y_train, cv=kf, scoring="r2")
cv_rmse_ridge = np.sqrt(-cross_val_score(pipe_ridge, X_train, y_train,
                                         cv=kf, scoring="neg_mean_squared_error"))

pipe_ridge.fit(X_train, y_train)
y_pred_ridge   = pipe_ridge.predict(X_test)
test_r2_ridge   = r2_score(y_test, y_pred_ridge)
test_rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print(f"  CV R²:   {cv_r2_ridge.mean():.4f} ± {cv_r2_ridge.std():.4f}")
print(f"  CV RMSE: {cv_rmse_ridge.mean():.4f} ± {cv_rmse_ridge.std():.4f}")
print(f"  Test R²:   {test_r2_ridge:.4f}")
print(f"  Test RMSE: {test_rmse_ridge:.4f}\n")

print("LASSO + Features Polinomiales")

ALPHAS_LASSO = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
lasso_scores = {}

for alpha in ALPHAS_LASSO:
    pipe = construir_pipeline(Lasso(alpha=alpha, max_iter=10000, random_state=SEED))
    scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    lasso_scores[alpha] = scores.mean()
    print(f"  alpha={alpha:<8} → CV R²={scores.mean():.4f}")

mejor_alpha_lasso = max(lasso_scores, key=lasso_scores.get)
print(f"\n Mejor alpha Lasso: {mejor_alpha_lasso} (CV R²={lasso_scores[mejor_alpha_lasso]:.4f})\n")

pipe_lasso = construir_pipeline(Lasso(alpha=mejor_alpha_lasso, max_iter=10000, random_state=SEED))
cv_r2_lasso   = cross_val_score(pipe_lasso, X_train, y_train, cv=kf, scoring="r2")
cv_rmse_lasso = np.sqrt(-cross_val_score(pipe_lasso, X_train, y_train,
                                         cv=kf, scoring="neg_mean_squared_error"))

pipe_lasso.fit(X_train, y_train)
y_pred_lasso   = pipe_lasso.predict(X_test)
test_r2_lasso   = r2_score(y_test, y_pred_lasso)
test_rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

print(f"  CV R²:   {cv_r2_lasso.mean():.4f} ± {cv_r2_lasso.std():.4f}")
print(f"  CV RMSE: {cv_rmse_lasso.mean():.4f} ± {cv_rmse_lasso.std():.4f}")
print(f"  Test R²:   {test_r2_lasso:.4f}")
print(f"  Test RMSE: {test_rmse_lasso:.4f}\n")

modelo_lasso = pipe_lasso.named_steps["model"]
n_activas = np.sum(modelo_lasso.coef_ != 0)
print(f"  Features activas tras Lasso polinomial: {n_activas}/{n_features_poly}\n")

poly_transformer = pipe_lasso.named_steps["poly"]
nombres_poly = poly_transformer.get_feature_names_out(features_disponibles)

coefs_lasso = pd.Series(modelo_lasso.coef_, index=nombres_poly)
top_interacciones = coefs_lasso[coefs_lasso != 0].abs().sort_values(ascending=False).head(10)

print("Términos más relevantes (Lasso polinomial):")
for nombre, val in top_interacciones.items():
    signo = "+" if coefs_lasso[nombre] > 0 else "-"
    print(f"  {signo} {nombre:<20} {abs(val):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Modelos Polinomiales (interacciones) — Predicción de PER NBA\n"
             f"Introducción al Aprendizaje Automático - UTFSM 2026",
             fontsize=12, fontweight="bold")

ax1 = axes[0]
ax1.scatter(y_test, y_pred_ridge, alpha=0.45, color=COLOR_RIDGE, s=20, edgecolors="none")
lims = [min(y_test.min(), y_pred_ridge.min())-1, max(y_test.max(), y_pred_ridge.max())+1]
ax1.plot(lims, lims, "k--", linewidth=1, label="Predicción perfecta")
ax1.set_xlabel("PER real (t+1)")
ax1.set_ylabel("PER predicho")
ax1.set_title(f"Ridge Polinomial (α={mejor_alpha_ridge})")
ax1.legend(fontsize=9)
ax1.grid(linestyle="--", alpha=0.3)
ax1.spines[["top","right"]].set_visible(False)
ax1.text(0.05, 0.92, f"R²={test_r2_ridge:.3f}\nRMSE={test_rmse_ridge:.3f}",
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

ax2 = axes[1]
ax2.scatter(y_test, y_pred_lasso, alpha=0.45, color=COLOR_LASSO, s=20, edgecolors="none")
ax2.plot(lims, lims, "k--", linewidth=1, label="Predicción perfecta")
ax2.set_xlabel("PER real (t+1)")
ax2.set_ylabel("PER predicho")
ax2.set_title(f"Lasso Polinomial (α={mejor_alpha_lasso})")
ax2.legend(fontsize=9)
ax2.grid(linestyle="--", alpha=0.3)
ax2.spines[["top","right"]].set_visible(False)
ax2.text(0.05, 0.92, f"R²={test_r2_lasso:.3f}\nRMSE={test_rmse_lasso:.3f}",
         transform=ax2.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

ax3 = axes[2]
bars = ax3.barh(top_interacciones.index[::-1], top_interacciones.values[::-1],
                color=COLOR_LASSO, alpha=0.8, edgecolor="white")
ax3.set_xlabel("|Coeficiente|")
ax3.set_title("Top 10 Términos — Lasso Polinomial")
ax3.grid(axis="x", linestyle="--", alpha=0.3)
ax3.spines[["top","right"]].set_visible(False)
ax3.tick_params(axis="y", labelsize=8)

plt.tight_layout()
ruta_fig = os.path.join(RUTA_OUTPUT, f"{NOMBRE}_resultados.png")
plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")

resultado = pd.DataFrame([
    {
        "Modelo": "Ridge_Polinomial",
        "Mejor alpha": mejor_alpha_ridge,
        "N features (con interacciones)": n_features_poly,
        "CV R² media": round(cv_r2_ridge.mean(), 4),
        "CV R² std": round(cv_r2_ridge.std(), 4),
        "CV RMSE media": round(cv_rmse_ridge.mean(), 4),
        "Test R²": round(test_r2_ridge, 4),
        "Test RMSE": round(test_rmse_ridge, 4),
    },
    {
        "Modelo": "Lasso_Polinomial",
        "Mejor alpha": mejor_alpha_lasso,
        "N features (con interacciones)": n_features_poly,
        "Features activas": int(n_activas),
        "CV R² media": round(cv_r2_lasso.mean(), 4),
        "CV R² std": round(cv_r2_lasso.std(), 4),
        "CV RMSE media": round(cv_rmse_lasso.mean(), 4),
        "Test R²": round(test_r2_lasso, 4),
        "Test RMSE": round(test_rmse_lasso, 4),
    },
])
resultado.to_csv(os.path.join(RUTA_OUTPUT, f"{NOMBRE}_metricas.csv"), index=False)
