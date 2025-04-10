import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# estilo general para las visualizaciones
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# carga del dataset limpio
def carga_revision(df=None):
    processed_path = "/Users/elena/Downloads/Data_science/proyectos/proyecto_I/interacciones-Facebook/data/dataset_limpio.csv"
    df = pd.read_csv(processed_path, sep=";")
    cat_columns = ['post_month', 'post_weekday', 'post_hour', 'paid']
    for col in cat_columns:
        df[col] = df[col].astype('category')
    return df

# muestra gráficas de distribución para variables categóricas
def distribucion_cat(df):
    for col in ['post_month', 'post_weekday', 'post_hour', 'paid', 'type']:
        sns.countplot(data=df, x=col)
        plt.title(f"Distribución de {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"outputs/distribucion_de_{col}.png")
        plt.close()
    return df

# muestra histogramas para variables numéricas relevantes
def distribucion_num(df):
    num_cols = ['like', 'comment', 'share', 'total_interactions',
                'lifetime_post_total_reach']
    for col in num_cols:
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribución de {col}")
        plt.tight_layout()
        plt.savefig(f"outputs/distribucion_de_{col}.png")
        plt.close()
    return df

# muestra boxplot de interacciones por día de la semana
def box_dia(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='post_weekday', y='total_interactions')
    plt.title("Interacciones por día de la semana")
    plt.xlabel("Día de la semana")
    plt.ylabel("Total de interacciones")
    plt.tight_layout()
    plt.savefig("outputs/intercciones_por_dia.png")
    plt.close()
    return df

# muestra boxplot de interacciones por tipo de publicación
def box_tipo(df):
    sns.boxplot(data=df, x='type', y='total_interactions')
    plt.title("Total de interacciones por tipo de publicación")
    plt.tight_layout()
    plt.savefig("outputs/intercciones_por_tipo.png")
    plt.close()
    return df

# muestra promedio de interaciionnes por hora
def media_hora(df):
    mean_interactions = df.groupby('post_hour')['total_interactions'].mean()
    sns.barplot(x=mean_interactions.index, y=mean_interactions.values)
    plt.title("Promedio de interacciones por hora")
    plt.xlabel("Hora")
    plt.ylabel("Interacciones promedio")
    plt.tight_layout()
    plt.savefig("outputs/promedio_intercciones_por_hora.png")
    plt.close()
    return df

# compara interacciones promedio entre publicaciones promocionadas y orgánicas
def media_paid(df):
    sns.barplot(data=df, x='paid', y='total_interactions', estimator='mean', ci='sd', palette='Set2')
    plt.title("Interacciones promedio según si el post fue promocionado")
    plt.xlabel("Promocionado (1 = sí, 0 = no)")
    plt.ylabel("Total de interacciones promedio")
    plt.xticks([0, 1], ['No promocionado', 'Promocionado'])
    plt.tight_layout()
    plt.savefig("outputs/promedio_intercciones_segun_promocion.png")
    plt.close()
    return df

# muestra la matriz de correlación entre variables numéricas
def matriz_corr(df):
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlación")
    plt.tight_layout()
    plt.savefig("outputs/matriz_correlacion.png")
    plt.close()
    return df

# función principal para ejecutar el análisis exploratorio completo
def run_eda_pipeline(df):
    df = None
    df = carga_revision(df)
    df = distribucion_cat(df)
    df = distribucion_num(df)
    df = box_dia(df)
    df = box_tipo(df)
    df = media_hora(df)
    df = media_paid(df)
    df = matriz_corr(df)
    return df