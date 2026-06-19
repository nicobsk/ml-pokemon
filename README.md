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

## Resultados

Se evaluaron dos modelos (KNN y Random Forest) bajo dos configuraciones de features: incluyendo o excluyendo el stat "Total" de cada pokemon.

| Modelo        | Features   | Accuracy | F1 Legendario | F1 CV         |
|---------------|------------|----------|---------------|---------------|
| KNN           | Con Total  | 96.11%   | 0.77          | 0.713 ± 0.057 |
| Random Forest | Con Total  | 98.33%   | 0.91          | 0.875 ± 0.036 |
| KNN           | Sin Total  | 95.56%   | 0.73          | 0.670 ± 0.064 |
| Random Forest | Sin Total  | 95.00%   | 0.69          | 0.601 ± 0.060 |

**Random Forest con todos los stats es el modelo ganador.**
No es sorpresa porque en el mundo pokemon, los legendarios se distinguen precisamente por tener stats base muy superiores al promedio, un ejemplo es Mewtwo con 680 de total no se parece en nada a un Rattata con 253.

El mayor desafío fue la clase minoritaria (legendarios representan ~8% del dataset), lo que hace que el modelo tienda a clasificar todo como "No Legendario" por defecto.

## Análisis de errores, Legendarios atípicos

El modelo tiende a fallar con un subconjunto de 8 legendarios cuyas stats base están muy por debajo del promedio de su categoría (613 de Total), siendo en algunos casos inferiores al promedio de un pokemon común (415):

| Pokémon          | Total |
|------------------|-------|
| Cosmog           | 200   |
| Meltan           | 300   |
| Kubfu            | 385   |
| Cosmoem          | 400   |
| Poipole          | 420   |
| Phione           | 480   |
| Zygarde 10%      | 486   |
| Calyrex          | 500   |

Desde el punto de vista del lore, esto tiene sentido: Cosmog es una nebulosa sin poder desarrollado, Kubfu es una cría de arte marcial, y Phione nunca fue reconocido oficialmente como legendario en los juegos. El modelo no los clasifica como legendarios por la misma razón que muchos entrenadores tampoco los consideran así, sus stats.

## Metodología

### Datos
El dataset utilizado contiene 1045 pokemon incluyendo formas alternativas, reducido a 897 tras una limpieza que eliminó:
- **Mega Evoluciones**: stats artificialmente infladas que no representan el estado base del pokemon.
- **Formas alternativas**: evitan duplicados por número de Pokédex (ej. Vulpix normal vs. Vulpix Alola).

La distribución final es de 805 no legendarios (89.7%) y 92 legendarios (10.3%), un desbalance de clases relevante que se abordó con `class_weight='balanced'` en Random Forest.

### Features
Se evaluaron dos configuraciones de features para comparar el impacto de la columna `Total`:
- **Con Total**: HP, Attack, Defense, Special Attack, Special Defense, Speed, Total.
- **Sin Total**: Las mismas 6 stats individuales sin el agregado.

Las columnas `Type`, `Other Type`, `Generation` y `Pokedex No.` fueron excluidas del modelo al ser categóricas o identificadoras no representativas del poder del Pokémon.

### Modelos
Se entrenaron y compararon dos pipelines:

**K-Nearest Neighbors (KNN)**
```python
Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])
```

**Random Forest**
```python
Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))])
```

Ambos incluyen `StandardScaler` para normalizar las stats antes de la clasificación, garantizando que ninguna variable domine por magnitud (ej. Total ~500 vs. Speed ~80).

### Validación
Se utilizó `StratifiedKFold` con 5 folds para preservar la proporción de legendarios en cada partición, dado el desbalance
de clases. La métrica principal de comparación fue el **F1-score** sobre la clase minoritaria (legendarios), complementado con
accuracy general. El dataset se dividió en 80% entrenamiento y 20% test con `random_state=42` para reproducibilidad.

### Experimento de generalización
Como prueba final, se construyó manualmente un dataframe con 15 pokemon exclusivamente de la Generación 9, no vista durante
el entrenamiento, para evaluar si el modelo es capaz de generalizar a nuevas generaciones. Se seleccionaron casos deliberadamente desafiantes: legendarios de stats bajas, no legendarios de stats altas y una forma alternativa en combate (Palafin Hero Form), para explorar los límites reales del modelo.

## Prueba con Pokémon de Generación 9

Para evaluar la capacidad de generalización del modelo a generaciones no vistas durante el entrenamiento, se construyó un dataframe manual con 15 pokemon de la Generación 9 (Escarlata y Púrpura), incluyendo
8 no legendarios y 7 legendarios, seleccionados deliberadamente para desafiar al modelo con casos límite.


### Resultados

| Pokémon            | Real          | Con Total     | Sin Total     |
|--------------------|---------------|---------------|---------------|
| Okidogi            | ¡Legendario!  | ❌ No Legend. | ❌ No Legend. |
| Munkidori          | ¡Legendario!  | ❌ No Legend. | ❌ No Legend. |
| Mewscarada         | No Legendario | ✅ No Legend. | ✅ No Legend. |
| Skeleridge         | No Legendario | ✅ No Legend. | ✅ No Legend. |
| Quaquaval          | No Legendario | ✅ No Legend. | ✅ No Legend. |
| Archaludon         | No Legendario | ❌ Legendario | ✅ No Legend. |
| Hydrapple          | No Legendario | ✅ No Legend. | ✅ No Legend. |
| Ogerpon Teal Mask  | ¡Legendario!  | ❌ No Legend. | ❌ No Legend. |
| Fezandipiti        | ¡Legendario!  | ❌ No Legend. | ❌ No Legend. |
| Baxcalibur         | No Legendario | ✅ No Legend. | ✅ No Legend. |
| Wo-Chien           | ¡Legendario!  | ✅ Legendario | ❌ No Legend. |
| Koraidon           | ¡Legendario!  | ✅ Legendario | ✅ Legendario |
| Miraidon           | ¡Legendario!  | ✅ Legendario | ✅ Legendario |
| Dondozo            | No Legendario | ✅ No Legend. | ✅ No Legend. |
| Palafin Hero Form  | No Legendario | ❌ Legendario | ❌ Legendario |

**Accuracy Con Total: 9/15 (60%)** 
**Accuracy Sin Total: 9/15 (60%)**

### Análisis de errores

**Falsos negativos (legendarios no detectados):**
Okidogi, Munkidori, Fezandipiti y Ogerpon son parte del nuevo paradigma de legendarios de la Gen 9, el juego los trata como legendarios por su rol narrativo, pero sus stats (555 y 550 de Total respectivamente) son comparables a las de pseudolegendarios como Garchomp o Metagross. El modelo nunca vio este patrón durante el entrenamiento, ya que los legendarios de generaciones anteriores casi siempre superan los 580 puntos. Wo-Chien, otro caso similar con 570 de Total, solo fue detectado correctamente con el modelo con Total, lo que sugiere que el umbral numérico SÍ importa.

**Falsos positivos (no legendarios clasificados como legendarios):**
Archaludon (600 de Total, solo en Con Total) y Palafin Hero Form (650 de Total, en ambos modelos) son los casos más interesantes.
Palafin es quizás el "engaño" más elegante del experimento: un delfín común que, al cambiar de forma en combate, alcanza stats dignas de un legendario. El modelo lo clasificó como legendario en ambos experimentos a ojos cerrados y desde un punto de vista puramente numérico, no se equivoca, pero tal como vemos con los casos de los legendarios, hay factores más allá de los numéricos para clasificar, por ejemplo este caso específico, que según el "Lore", Palafin hace cameo a los superheroes de cómics que llega a la batalla, sale de escena y cuando nadie lo ve (al cambiarlo por otro pokemon), vuelve a la batalla y activa su habilidad para mostrar su forma Heroe.

### Conclusión

El modelo generaliza bien para legendarios "clásicos" de stats altas (Koraidon, Miraidon) y para no legendarios típicos, pero falla con la nueva generación de legendarios "narrativos" cuyo poder no se refleja en sus números. Esto no es un defecto del modelo sino una consecuencia del diseño del juego: Game Freak está redefiniendo qué significa ser legendario, y los datos
históricos aún no capturan ese cambio. Aun queda camino por recorrer, y quizá con dataframes más grandes, podría cometer menos errores, por ahora no pasa de un ejercicio interactivo entretenido, pero para gente que crea romhacks o fangames, podría ser un indicador (ignorando elementos narrativos) si definir su pokemon nuevo (fakemon) como un legendario o no.
