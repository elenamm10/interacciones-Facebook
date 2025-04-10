# 📊 Proyecto de Ciencia de Datos: Análisis de Publicaciones de Facebook

Este proyecto forma parte del Máster en Data Science y tiene como objetivo principal aplicar un flujo de trabajo de análisis de datos sobre un conjunto real de publicaciones de Facebook de una marca cosmética. Desde la limpieza de datos hasta la modelización predictiva, se busca responder a la pregunta: “¿Qué factores influyen en la viralidad de una publicación en Facebook de una marca de cosméticos?”

---

## 🎯 Preguntas de investigación

1. ¿Qué factores están asociados al aumento de interacciones en publicaciones de Facebook?
2. ¿Qué diferencias hay entre publicaciones promocionadas y orgánicas?
3. ¿Qué franjas horarias y días generan más viralidad?
4. ¿Existen patrones que permitan predecir si un post será viral?
5. ¿Qué tipo de publicación (Foto, Vídeo, Link, Estado) tiene mejor rendimiento?

---

## 📁 Estructura del Repositorio

```
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
├── notebooks/
    ├── etl.ipynb       # Extracción, transformación y limpieza de datos
    ├── eda.ipynb       # Análisis exploratorio de datos y visualizaciones
    ├── stats.ipynb     # Estadística inferencial y modelado predictivo
├── src/
    ├── etl.py          # Versión funcional del proceso ETL
    ├── eda.py          # Funciones gráficas y de exploración
    ├── stats.py        # Funciones de contraste de hipótesis y modelos
├── main.py
├── requirements.txt
└── README.md
```

---

## 📘 Descripción del Dataset

- 📄 500 publicaciones de Facebook
- 🔁 Métricas de rendimiento: likes, comentarios, shares, alcance, viralidad
- 📆 Información temporal: mes, día de la semana, hora
- 📣 Variable "paid": indica si el post fue promocionado
- 🎯 Variable target creada: "viral" (supera el percentil 95 de interacciones)

---

## 🔄 ETL — Limpieza y Transformación

📍 Script: [`etl.py`](src/etl.py)

- Renombrado de columnas
- Conversión de tipos de datos
- Creación de variable `post_period` (mañana, tarde, noche…)
- Imputación de valores nulos
- Definición de publicaciones virales (percentil 95)
- Guardado de datos en `/data/processed`

---

## 📊 EDA — Análisis Exploratorio

📍 Script: [`eda.py`](src/eda.py)

- Distribuciones de tipo de post, franja horaria, día
- Comparativa entre publicaciones virales y no virales
- Medias por grupo
- Identificación y análisis de outliers virales
- Gráficos exportados a `/outputs`

---

## 🧪 Análisis Estadístico

📍 Script: [`stats.py`](src/stats.py)

### Contrastes de hipótesis

- **Shapiro-Wilk**: normalidad entre grupos
- **Levene**: homogeneidad de varianzas
- **Welch’s T-test**: interacción en promocionadas vs orgánicas
- **Mann-Whitney U**: test no paramétrico
- **Kruskal-Wallis**: diferencias por día y tipo
- **Test de Dunn**: post-hoc para días y tipos

---

## 🤖 Modelado Predictivo

### Regresión Logística (penalizada y balanceada)

- Recall en publicaciones virales: **100%**
- Precisión: baja (clasifica muchos falsos positivos)
- Utilidad: alto valor de sensibilidad

### Árbol de Decisión

- Max depth = 4
- Reglas interpretables:
  - Si tipo = Foto y alto alcance → probable viralidad
  - Si tipo = Link y poco alcance → muy baja probabilidad

---

## 📈 Visualización y Dashboard

- Se ha creado un dashboard en **Power BI** con:
  - Distribución de publicaciones
  - Comparativas por tipo y viralidad
  - Árbol de decisión interpretativo
  - Métricas clave: número de virales, media de interacciones

---

## 🧠 Conclusiones

- Las publicaciones tipo **Foto** y con **alto alcance** tienen mayor probabilidad de ser virales.
- Las publicaciones **promocionadas** no garantizan mejores resultados, pero muestran mayor variabilidad.
- La **tarde-noche** y los días laborables parecen más efectivos para publicar.
- Los **modelos predictivos** muestran que es posible anticipar si un post será viral.
- La **regresión logística penalizada** es útil para no perder casos virales (alto recall), y el árbol de decisión aporta interpretabilidad.

---

## ⚙️ Ejecución del proyecto

```bash
pip install -r requirements.txt
python main.py
```

---

## 👩‍💻 Autoría

Trabajo realizado por Elena Millán como parte del Máster en Data Science e IA. Proyecto orientado a la integración de procesos de análisis, visualización y modelado en un flujo completo de datos reales.

---