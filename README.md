# Clasificación de Pokémon Legendarios mediante Machine Learning

## Integrantes

* Alonso Díaz
* Nicolás Mena

## Definición del Problema

Este proyecto busca responder las siguientes preguntas:

1. ¿Se puede predecir si un Pokémon es legendario utilizando únicamente sus estadísticas base?
2. Dentro del grupo de Pokémon con estadísticas altas, ¿es posible distinguir cuáles son realmente legendarios y cuáles no?

Estas preguntas corresponden a un problema de clasificación binaria, donde la variable objetivo es `Legendary` (0 = No Legendario, 1 = Legendario). El desafío surge porque existen Pokémon no legendarios con estadísticas comparables a las de algunos legendarios, lo que dificulta resolver el problema mediante reglas simples.

---

## Plan de Acción

### Dataset

Se utilizó el dataset **Pokemon Gen1-Gen8** obtenido desde Kaggle.

Características principales:

* 898 Pokémon pertenecientes a las generaciones 1 a 8.
* 13 variables relacionadas con estadísticas, tipos, generación y condición legendaria.
* Variable objetivo: `Legendary`.

Como parte del preprocesamiento se realizaron las siguientes tareas:

* Tratamiento de valores nulos en la columna `Other Type`.
* Eliminación de Mega Evoluciones para evitar sesgos en las estadísticas.
* Eliminación de formas alternativas para trabajar únicamente con formas base.
* Análisis exploratorio de datos (EDA) mediante gráficos descriptivos y análisis de correlación.

### Modelos Seleccionados

Para abordar el problema se seleccionaron dos algoritmos de clasificación:

* K-Nearest Neighbors (KNN)
* Random Forest

Además, se diseñaron dos experimentos:

* Utilizando las estadísticas individuales junto con la variable `Total`.
* Utilizando únicamente las estadísticas individuales.

### Estrategia de Evaluación

Se realizará una división estratificada de los datos en entrenamiento y prueba (80/20).

Posteriormente, los modelos serán evaluados mediante validación cruzada estratificada (`StratifiedKFold`) para estimar su capacidad de generalización y comparar su desempeño.

---

## Justificación de los Modelos

### K-Nearest Neighbors (KNN)

KNN permite clasificar observaciones según la similitud de sus características. Se considera adecuado para este problema porque permite analizar si los Pokémon legendarios forman grupos con patrones estadísticos similares.

Debido a que el algoritmo es sensible a la escala de los datos, se utiliza normalización mediante `StandardScaler`.

### Random Forest

Random Forest fue seleccionado por su capacidad para identificar relaciones complejas entre variables y por su buen desempeño en problemas de clasificación con datos tabulares.

Además, se configuró utilizando balanceo de clases (`class_weight='balanced'`) para compensar la menor cantidad de Pokémon legendarios presentes en el dataset.
