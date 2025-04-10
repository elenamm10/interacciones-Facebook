import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, kruskal
from statsmodels.formula.api import ols, logit
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, plot_tree



# --- CONTRASTES DE HIPÓTESIS ---

# comparativa entre posts promocionados y orgánicos
def contraste_paid(df):
    promo = df[df["paid"] == 1]["total_interactions"]
    no_promo = df[df["paid"] == 0]["total_interactions"]

    print("Shapiro (promocionadas):", shapiro(promo))
    print("Shapiro (no promocionadas):", shapiro(no_promo))
    print("Levene (homogeneidad de varianza):", levene(promo, no_promo))
    
    t_stat, p_val = ttest_ind(promo, no_promo, equal_var=False)
    print(f"Welch T-test: t = {t_stat:.4f}, p = {p_val:.4f}")
    
    stat, p = mannwhitneyu(promo, no_promo, alternative='two-sided')
    print(f"Mann-Whitney U: U = {stat}, p = {p:.4f}")
    return df

# Shapiro-Wilk por día de la semana
def sw_dia(df):
    for day in df["post_weekday"].unique():
        sub = df[df["post_weekday"] == day]["total_interactions"]
        _, p = shapiro(sub)
        print(f"Día {day} - Shapiro p-value: {p:.4f}")
    return df

# Levene por día de la semana
def levene_dia(df):
    groups = [df[df["post_weekday"] == day]["total_interactions"] for day in df["post_weekday"].unique()]
    stat, p = levene(*groups)
    print(f"Levene entre días - p-value: {p:.4f}")
    return df

# Kruskal-Wallis por día
def kw_dia(df):
    grouped = [group["total_interactions"].values for name, group in df.groupby("post_weekday")]
    stat, p = kruskal(*grouped)
    print(f"Kruskal-Wallis: H = {stat:.4f}, p = {p:.4f}")
    return df

# Test de Dunn post hoc
def dun_dia(df):
    dunn = sp.posthoc_dunn(df, val_col="total_interactions", group_col="post_weekday", p_adjust="holm")
    print("Test de Dunn (p-values ajustados):")
    print(dunn.round(4))
    dunn.to_csv("outputs/dunn_posthoc.csv")
    return df



# --- PREPROCESADO Y MODELADO ---

# nos aseguramos de que viral esté en formato entero (0 y 1)
def viral_ent(df):
    df["viral"] = df["viral"].astype(int)
    return df

# definimos variables independientes y dependiente
def definicion(df):
    global features, target, X, y
    features = ["type", "paid", "post_period", "post_weekday", "lifetime_post_total_reach"]
    target = "viral"
    X = df[features]
    y = df[target]
    return df


# separar variables categóricas y numéricas y crear preprocesador
def preprocesador(df):
    global categorical, numerical, preprocessor
    categorical = ["type", "paid", "post_period", "post_weekday"]
    numerical = ["lifetime_post_total_reach"]
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical),
        ("num", StandardScaler(), numerical)
    ])
    return df


# división en conjunto de entrenamiento y test
def entrenamiento_prueba(df):
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return df

# entrenamiento del modelo de regresión logística con class_weight balanced
def modelo(df):
    global logit_pipeline
    logit_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    logit_pipeline.fit(X_train, y_train)
    return df

# evaluación del modelo logístico: matriz de confusión y métricas
def evaluacion(df):
    y_pred = logit_pipeline.predict(X_test)
    print("Evaluación regresión logística:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    return df


# curva ROC
def curva_roc(df):
    y_prob = logit_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Curva ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/roc_logistica.png")
    plt.close()
    return df


# entrenar árbol de decisión sobre los mismos datos
def arbol_train(df):
    global tree_pipeline
    tree_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42))
    ])
    tree_pipeline.fit(X_train[features], y_train)
    return df

# árbol de decisión
def arbol(df):
    plt.figure(figsize=(18, 10))
    plot_tree(
        tree_pipeline.named_steps["classifier"],
        feature_names=tree_pipeline.named_steps["preprocessing"].get_feature_names_out(),
        class_names=["No viral", "Viral"],
        filled=True,
        rounded=True
    )
    plt.title("Árbol de decisión")
    plt.tight_layout()
    plt.savefig("outputs/arbol_decision.png")
    plt.close()
    return df

# función para ejecutar el pipeline de estadísticas
def run_stats_pipeline(df):
    df = contraste_paid(df)
    df = sw_dia(df)
    df = levene_dia(df)
    df = kw_dia(df)
    df = dun_dia(df)
    df = viral_ent(df)
    df = definicion(df)
    df = preprocesador(df)
    df = entrenamiento_prueba(df)
    df = modelo(df)
    df = evaluacion(df)
    df = curva_roc(df)
    df = arbol_train(df)
    df = arbol(df)
    return df