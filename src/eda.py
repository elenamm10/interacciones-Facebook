import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo general para las visualizaciones
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(path="data/processed/dataset_limpio.csv"):
    """carga el dataset limpio"""
    df = pd.read_csv(path, sep=";")
    cat_columns = ['post_month', 'post_weekday', 'post_hour', 'paid']
    for col in cat_columns:
        df[col] = df[col].astype('category')
    return df

def plot_categorical_distributions(df):
    """muestra gráficas de distribución para variables categóricas"""
    for col in ['post_month', 'post_weekday', 'post_hour', 'paid', 'type']:
        sns.countplot(data=df, x=col)
        plt.title(f"Distribución de {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def plot_numerical_distributions(df):
    """muestra histogramas para variables numéricas relevantes"""
    num_cols = ['like', 'comment', 'share', 'total_interactions',
                'lifetime_post_total_reach', 'lifetime_post_total_impressions',
                'lifetime_engaged_users', 'lifetime_post_consumers',
                'lifetime_post_consumptions']
    for col in num_cols:
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribución de {col}")
        plt.tight_layout()
        plt.show()

def analyze_viral_posts(df):
    """añade columna 'viral' y compara proporciones por grupo"""
    threshold = df['total_interactions'].quantile(0.95)
    df['viral'] = df['total_interactions'] > threshold
    summary = df.groupby('viral')[['type', 'post_hour', 'post_weekday', 'paid']].agg(
        lambda x: x.value_counts(normalize=True).to_dict()
    )
    return summary

def plot_interactions_by_weekday(df):
    """genera un boxplot de las interacciones en función del día de la semana"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='post_weekday', y='total_interactions')
    plt.title("Interacciones por día de la semana")
    plt.xlabel("Día de la semana")
    plt.ylabel("Total de interacciones")
    plt.tight_layout()
    plt.show()

def plot_interactions_by_type(df):
    """muestra boxplot de interacciones por tipo de publicación"""
    sns.boxplot(data=df, x='type', y='total_interactions')
    plt.title("Total de interacciones por tipo de publicación")
    plt.tight_layout()
    plt.show()

def plot_avg_interactions_by_hour(df):
    """muestra promedio de interacciones por hora"""
    mean_interactions = df.groupby('post_hour')['total_interactions'].mean()
    sns.barplot(x=mean_interactions.index, y=mean_interactions.values)
    plt.title("Promedio de interacciones por hora")
    plt.xlabel("Hora")
    plt.ylabel("Interacciones promedio")
    plt.tight_layout()
    plt.show()

def plot_promoted_vs_organic(df):
    """compara interacciones promedio entre publicaciones promocionadas y orgánicas"""
    sns.barplot(data=df, x='paid', y='total_interactions', estimator='mean', ci='sd', palette='Set2')
    plt.title("Interacciones promedio según si el post fue promocionado")
    plt.xlabel("Promocionado (1 = sí, 0 = no)")
    plt.ylabel("Total de interacciones promedio")
    plt.xticks([0, 1], ['No promocionado', 'Promocionado'])
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """muestra la matriz de correlación entre variables numéricas"""
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlación")
    plt.tight_layout()
    plt.show()

from src.eda import *

df = load_data()

plot_categorical_distributions(df)
plot_numerical_distributions(df)

print(analyze_viral_posts(df))

plot_interactions_by_type(df)
plot_avg_interactions_by_hour(df)
plot_promoted_vs_organic(df)
plot_correlation_matrix(df)