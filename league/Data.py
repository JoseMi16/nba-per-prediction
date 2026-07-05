import os
import pandas as pd
import numpy as np


RUTA_BASE    = "./league"
RUTA_KAGGLE  = "./league/Kaggle"
RUTA_OUTPUT  = "./output"
os.makedirs(RUTA_OUTPUT, exist_ok=True)

TEMPORADAS = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

MIN_PARTIDOS = 20

lista_advanced = []

for t in TEMPORADAS:
    archivo = os.path.join(RUTA_BASE, f"Nba_advanced_{t:02d}.csv")
    if not os.path.exists(archivo):
        print(f" No encontrado: {archivo}")
        continue

    df = pd.read_csv(archivo)

    df["season"] = 2000 + t

    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"]
        df = df.drop(columns=["Rk"], errors="ignore")

    lista_advanced.append(df)

df_advanced = pd.concat(lista_advanced, ignore_index=True)



lista_pergame = []

for t in TEMPORADAS:
    archivo = os.path.join(RUTA_BASE, f"NBA_PerGame_{t:02d}.csv")
    if not os.path.exists(archivo):
        continue

    df = pd.read_csv(archivo)
    df["season"] = 2000 + t

    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"]
        df = df.drop(columns=["Rk"], errors="ignore")
    lista_pergame.append(df)

df_pergame = pd.concat(lista_pergame, ignore_index=True)

kaggle_advanced_path  = os.path.join(RUTA_KAGGLE, "Advanced.csv")
kaggle_pergame_path   = os.path.join(RUTA_KAGGLE, "Player Per Game.csv")

df_kaggle_adv     = pd.read_csv(kaggle_advanced_path)  if os.path.exists(kaggle_advanced_path)  else pd.DataFrame()
df_kaggle_pergame = pd.read_csv(kaggle_pergame_path)   if os.path.exists(kaggle_pergame_path)    else pd.DataFrame()

COLS_ADV = ["Player", "season", "Pos", "Age", "G", "MP",
            "PER", "TS%", "USG%", "BPM", "VORP", "WS", "OWS", "DWS"]

COLS_PG  = ["Player", "season", "PTS", "TRB", "AST", "STL", "BLK", "TOV", "FG%", "3P%", "FT%"]

def normalizar_columnas(df, cols_requeridas, nombre):
    cols_disponibles = [c for c in cols_requeridas if c in df.columns]
    cols_faltantes   = [c for c in cols_requeridas if c not in df.columns]
    if cols_faltantes:
        print(f"{nombre},{cols_faltantes}")
    df = df[cols_disponibles].copy()
    for c in cols_disponibles:
        if c not in ["Player", "Pos"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df_advanced = normalizar_columnas(df_advanced, COLS_ADV, "Advanced")
df_pergame  = normalizar_columnas(df_pergame,  COLS_PG,  "PerGame")

if "Tm" in df_advanced.columns:
    df_advanced = (df_advanced
                   .sort_values("Tm", key=lambda x: x == "TOT", ascending=False)
                   .drop_duplicates(subset=["Player", "season"], keep="first"))
else:
    df_advanced = df_advanced.drop_duplicates(subset=["Player", "season"], keep="first")

if "Tm" in df_pergame.columns:
    df_pergame = (df_pergame
                  .sort_values("Tm", key=lambda x: x == "TOT", ascending=False)
                  .drop_duplicates(subset=["Player", "season"], keep="first"))
else:
    df_pergame = df_pergame.drop_duplicates(subset=["Player", "season"], keep="first")

antes = len(df_advanced)
df_advanced = df_advanced[df_advanced["G"] >= MIN_PARTIDOS]


df_merged = pd.merge(df_advanced, df_pergame, on=["Player", "season"], how="left")

df_merged = df_merged.sort_values(["Player", "season"])


df_merged["PER_next"] = (df_merged
                         .groupby("Player")["PER"]
                         .shift(-1))

df_final = df_merged.dropna(subset=["PER_next", "PER"]).copy()

FEATURES_CLAVE = ["PER", "TS%", "USG%", "BPM", "PTS", "TRB", "AST"]
antes = len(df_final)
df_final = df_final.dropna(subset=[f for f in FEATURES_CLAVE if f in df_final.columns])

ruta_final = os.path.join(RUTA_OUTPUT, "nba_dataset_ml.csv")
df_final.to_csv(ruta_final, index=False)

FEATURES_MODELO = [c for c in ["PER", "TS%", "USG%", "BPM", "VORP", "WS",
                                "OWS", "DWS", "PTS", "TRB", "AST", "STL",
                                "BLK", "TOV", "Age", "G", "MP"] if c in df_final.columns]

df_modelo = df_final[["Player", "season"] + FEATURES_MODELO + ["PER_next"]].copy()
ruta_modelo = os.path.join(RUTA_OUTPUT, "nba_features_target.csv")
df_modelo.to_csv(ruta_modelo, index=False)

ruta_resumen = os.path.join(RUTA_OUTPUT, "resumen_estadistico.csv")
df_modelo[FEATURES_MODELO + ["PER_next"]].describe().to_csv(ruta_resumen)
