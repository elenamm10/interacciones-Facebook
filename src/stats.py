import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, kruskal
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


def cargar_datos(path):
    return pd.read_csv(path)


def prueba_normalidad_por_grupo(df, group_col, value_col):
    resultados = {}
    for g in df[group_col].unique():
        datos = df[df[group_col] == g][value_col]
        stat, p = shapiro(datos)
        resultados[g] = p
    return resultados


def prueba_levene(df, group_col, value_col):
    grupos = [df[df[group_col] == g][value_col] for g in df[group_col].unique()]
    return levene(*grupos)


def prueba_kruskal(df, group_col, value_col):
    grupos = [df[df[group_col] == g][value_col] for g in df[group_col].unique()]
    return kruskal(*grupos)


def prueba_posthoc_dunn(df, group_col, value_col):
    from scikit_posthocs import posthoc_dunn
    return posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust='bonferroni')


def entrenar_regresion_logistica(X_train, y_train):
    categorical = ['type', 'paid', 'post_period', 'post_weekday']
    numerical = ['lifetime_post_total_reach']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first'), categorical),
        ('num', StandardScaler(), numerical)
    ])

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(
            penalty='l2', C=1.0, max_iter=1000, solver='lbfgs', class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=4)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    return cm, cr, fpr, tpr, roc_auc


def visualizar_curva_roc(fpr, tpr, roc_auc):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Curva ROC - Regresión logística (balanced)")
    plt.legend()
    plt.show()


def entrenar_arbol_decision(X_train, y_train, selected_features):
    cat_features = [f for f in selected_features if X_train[f].dtype == 'object' or X_train[f].dtype == 'bool']
    num_features = [f for f in selected_features if f not in cat_features]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ])

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
    ])

    pipeline.fit(X_train[selected_features], y_train)
    return pipeline


def visualizar_arbol_decision(model):
    plt.figure(figsize=(20, 10))
    plot_tree(
        model.named_steps['classifier'],
        feature_names=model.named_steps['preprocessing'].get_feature_names_out(),
        class_names=['no viral', 'viral'],
        filled=True, rounded=True
    )
    plt.title("Árbol de decisión")
    plt.show()