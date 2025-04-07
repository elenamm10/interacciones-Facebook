import pandas as pd

def load_data(raw_path):
    """carga el dataset original desde una ruta dada"""
    return pd.read_csv(raw_path, sep=';')

def clean_columns(df):
    """limpia los nombres de las columnas (minúsculas, guiones bajos, sin espacios)"""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def transform_data(df):
    """elimina columnas irrelevantes y cambia tipos de variables a categóricas"""
    if 'category' in df.columns:
        df = df.drop(columns=['category'])
    
    categorical = ['post_month', 'post_weekday', 'post_hour', 'paid']
    for col in categorical:
        df[col] = df[col].astype('category')
    
    return df

def handle_missing_values(df):
    """gestiona los valores nulos imputando """
    # Imputamos 'paid' como 0.0 (orgánica) por ser la más común
    df['paid'] = df['paid'].fillna(0.0)
    
    # Imputamos 'like' y 'share' como 0 por coherencia con interacciones
    df['like'] = df['like'].fillna(0)
    df['share'] = df['share'].fillna(0)
    
    return df

def save_data(df, processed_path):
    """guarda el dataset limpio en la ruta especificada"""
    df.to_csv(processed_path, sep=';', index=False)

def main():
    raw_path = "data/raw/dataset_Facebook.csv"
    processed_path = "data/processed/dataset_limpio.csv"

    df = load_data(raw_path)
    df = clean_columns(df)
    df = transform_data(df)
    df = handle_missing_values(df)
    save_data(df, processed_path)
    print(f"Dataset limpio guardado en: {processed_path}")

if __name__ == "__main__":
    main()