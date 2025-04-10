from etl import run_etl_pipeline
from eda import run_eda_pipeline
from stats import run_stats_pipeline

def main():
    print("Iniciando pipeline completo del proyecto...")

    # Paso 1: ETL - Limpieza y transformación
    print("\n Ejecutando ETL...")
    df = run_etl_pipeline()

    # Paso 2: Análisis Exploratorio
    print("\n Ejecutando EDA...")
    df = run_eda_pipeline(df)

    # Paso 3: Análisis Estadístico y Modelado
    print("\n Ejecutando análisis estadístico y modelado...")
    df = run_stats_pipeline(df)

    print("\n Pipeline completado con éxito.")

if __name__ == "__main__":
    main()