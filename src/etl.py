import pandas as pd

# carga del dataset original
def carga(df=None):
    raw_path = "/Users/elena/Downloads/Data_science/proyectos/proyecto_I/interacciones-Facebook/data/dataset_Facebook.csv"
    df = pd.read_csv(raw_path, sep=';')
    return df

# limpia los nombres de las columnas (minúsculas, guiones bajos, sin espacios)
def limpia_col(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# elimina columnas irrelevantes
def borra_col(df):
    if 'category' in df.columns:
        df = df.drop(columns=['category'])
    return df

# crear variable post_period según hora
def var_nueva(df):
    def get_post_period(hour):
        if 6 <= hour < 12:
            return "mañana"
        elif 12 <= hour < 18:
            return "tarde"
        elif 18 <= hour < 24:
            return "tarde_noche"
        else:
            return "noche"
    df["post_period"] = df["post_hour"].apply(get_post_period)
    return df

# convierte columnas a categóricas
def cambio_tipo(df):
    categorical = ['post_month', 'post_weekday', 'post_hour', 'paid']
    for col in categorical:
        df[col] = df[col].astype('category')
    return df

# gestión de valores nulos
def nulos(df):
    # omputamos 'paid' como 0.0 (orgánica) por ser la más común
    df['paid'] = df['paid'].fillna(0.0)
    
    # imputamos 'like' y 'share' como 0 por coherencia con interacciones
    df['like'] = df['like'].fillna(0)
    df['share'] = df['share'].fillna(0)
    
    return df

# crea variable objetivo: viralidad por percentil 95
def var_viral(df):
    p95 = df["total_interactions"].quantile(0.95)
    df["viral"] = (df["total_interactions"] > p95).astype(bool)
    return df

# guarda el dataset limpio en la ruta especificada
def guardado(df):
    processed_path = "/Users/elena/Downloads/Data_science/proyectos/proyecto_I/interacciones-Facebook/data/dataset_limpio.csv"
    df.to_csv(processed_path, sep=';', index=False)

# función principal que ejecuta el pipeline ETL
def run_etl_pipeline():
    df = None
    df = carga(df)
    df = limpia_col(df)
    df = borra_col(df)
    df = var_nueva(df)
    df = cambio_tipo(df)
    df = nulos(df)
    df = var_viral(df)
    df = guardado(df)
    return df