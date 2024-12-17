# Desafíos de Procesamiento de Lenguaje Natural

Este repositorio contiene los archivos, códigos y análisis correspondientes a los cuatro desafíos realizados en la materia de **Procesamiento de Lenguaje Natural (PLN)**.

---

## Desafío 1: Vectorización de texto y clasificación con Naïve Bayes usando el dataset 20 Newsgroups

### Descripción general:

Este práctico tiene como objetivo trabajar con **vectorización de texto**, calcular **similitudes de documentos** y entrenar modelos de clasificación **Naïve Bayes** utilizando el clásico conjunto de datos **20 Newsgroups**. Se abordan técnicas de **TF-IDF** para la representación de documentos y se optimizan modelos de clasificación para maximizar el desempeño en términos del **F1-Score macro**.

### Contenidos del práctico:

1. Carga y preprocesamiento de datos:
   - Se utiliza el dataset **20 Newsgroups** incluido en `sklearn`, eliminando encabezados, pies y citas.
   - Los datos se separan en conjuntos de **entrenamiento** y **test**.

2. Vectorización de los textos:
   - Se implementa la vectorización mediante **TF-IDF (Term Frequency - Inverse Document Frequency)**.
   - La matriz obtenida se interpreta como **matriz documento-término**.

3. Cálculo de similitud entre documentos:
   - Se calcula la **similaridad coseno** entre un documento dado y el resto de los documentos.
   - Se seleccionan documentos al azar y se determinan los **5 documentos más similares**.

4. Entrenamiento de modelos Naïve Bayes:
   - Se entrenan modelos de clasificación **MultinomialNB** y **ComplementNB** utilizando `GridSearchCV` para optimizar hiperparámetros.
   - Se evalúa el desempeño de los modelos en el conjunto de **test** utilizando el **F1-Score macro**.

5. Matriz término-documento y similitud entre palabras:
   - Se transpone la matriz documento-término para obtener una **matriz término-documento**.
   - Se seleccionan manualmente **5 palabras** relevantes (ej., "computer", "politics", etc.) y se calculan sus **5 palabras más similares** mediante **similaridad coseno**.

### Librerías utilizadas:
- **Python 3.9+**
- **Scikit-learn** (para carga de datos, vectorización, modelos y evaluación)
- **Numpy** (para operaciones con matrices y manipulación de datos)

## Desafío 2: Vectores de palabras con Gensim y visualización

### Descripción general:

Este práctico tiene como objetivo trabajar con **modelos de embeddings de palabras**, utilizando la librería **Gensim** para crear representaciones vectoriales de las palabras del texto de la novela *Orgullo y Prejuicio* de Jane Austen. A partir de los embeddings generados, se analizan las **similitudes semánticas** entre términos clave y se visualiza su distribución en un espacio reducido (2D y 3D) para entender mejor las relaciones entre las palabras.

### Contenidos del práctico:

1. **Carga y preprocesamiento de datos:**
   - Se utiliza el texto completo de la novela *Orgullo y Prejuicio*, cargado desde un archivo de texto.
   - Se realiza el procesamiento de texto, dividiendo el contenido en oraciones y luego en tokens (palabras) utilizando la función `text_to_word_sequence` de `Keras`.

2. **Creación de embeddings de palabras con Gensim:**
   - Se entrena un modelo de **Word2Vec** utilizando la librería **Gensim** sobre el texto de la novela.
   - Se emplea el enfoque **skipgram** para generar los vectores de palabras, configurando parámetros como el tamaño de la ventana, la dimensionalidad del espacio vectorial y la frecuencia mínima de palabras.

3. **Análisis de similitudes entre palabras:**
   - Se calculan las **similitudes semánticas** entre palabras relevantes del texto, como “matrimonio”, “orgullo” y “honor”.
   - Se realizan análisis de **analogías** como "Elizabeth + Bingley - Darcy" para explorar las relaciones entre los personajes y conceptos.

4. **Reducción de dimensionalidad y visualización:**
   - Se reduce la dimensionalidad de los vectores de palabras generados utilizando **Incremental PCA** para proyectarlos en un espacio de 2D y 3D.
   - Se visualizan los embeddings en gráficos interactivos con **Plotly**, permitiendo explorar las relaciones semánticas entre las palabras en un espacio reducido.

5. **Conclusiones:**
   - Se observa cómo el modelo captura las relaciones semánticas entre los personajes y conceptos clave de la novela, como el matrimonio, la familia y el honor.
   - Las visualizaciones muestran cómo las palabras relacionadas se agrupan en el espacio vectorial, reflejando las dinámicas sociales y los temas centrales de la obra.

### Librerías utilizadas:
- **Python 3.9+**
- **Gensim** (para la creación de embeddings de palabras con Word2Vec)
- **Keras** (para preprocesamiento y tokenización del texto)
- **Scikit-learn** (para reducción de dimensionalidad con PCA)
- **Plotly** (para la visualización interactiva de los vectores de palabras en 2D y 3D)

## Desafío 3: Modelos de Lenguaje con RNNs

### Descripción general:

Este práctico tiene como objetivo implementar un modelo de lenguaje basado en una Red Neuronal Recurrente (RNN) utilizando **LSTM (Long Short-Term Memory)** para generar texto a partir de un corpus. El proyecto incluye la tokenización de texto, el entrenamiento de un modelo de lenguaje y la evaluación del rendimiento utilizando técnicas de generación de texto como **Greedy Search** y **Beam Search**.

### Contenidos del práctico:

1. **Preprocesamiento de los datos:**
   - El corpus de texto utilizado es **"El Fantasma de Canterville"** de Oscar Wilde, cargado desde un archivo `.txt`.
   - Se realiza la limpieza del texto eliminando caracteres no deseados y segmentando el corpus en oraciones.
   - El texto se tokeniza utilizando la clase **Tokenizer** de Keras y se convierte en secuencias numéricas.
   - Se divide el conjunto de datos en secuencias de longitud fija y se aplica **padding** a las secuencias.

2. **Modelo de lenguaje con RNN:**
   - Se construye un modelo **LSTM** utilizando Keras, donde la entrada es una secuencia de palabras y la salida es la predicción de la siguiente palabra en la secuencia.
   - Se entrena el modelo utilizando **categorical crossentropy** como función de pérdida y **adam** como optimizador.

3. **Generación de texto:**
   - Se implementan dos técnicas para generar texto a partir del modelo entrenado:
     - **Greedy Search:** el modelo selecciona la palabra con la mayor probabilidad en cada paso.
     - **Beam Search:** se consideran múltiples opciones para cada palabra generada y se escoge la secuencia más probable.
   
4. **Evaluación del modelo:**
   - Se evalúa el modelo utilizando la **perplejidad** como métrica de desempeño, lo que permite medir la calidad de las predicciones generadas por el modelo.

5. **Resultados y visualización:**
   - Se presentan ejemplos de texto generado por el modelo utilizando ambas estrategias de generación.
   - Se visualizan las curvas de entrenamiento y validación para observar el desempeño durante el entrenamiento.

### Librerías utilizadas:
- **Python 3.9+**
- **TensorFlow / Keras** (para la construcción, entrenamiento y evaluación del modelo de redes neuronales)
- **Numpy** (para manipulación de datos numéricos)
- **Matplotlib / Seaborn** (para visualización de resultados)
- **Scikit-learn** (para preprocesamiento y evaluación de métricas)

## Desafío 4: Chatbot de Preguntas y Respuestas con LSTM utilizando el Dataset ConvAI2

### Descripción general:

Este práctico tiene como objetivo construir un **Chatbot de Preguntas y Respuestas (QA)** utilizando una red neuronal recurrente **Long Short-Term Memory (LSTM)**. El modelo se adapta a un conjunto de datos conversacionales para generar respuestas a preguntas formuladas en inglés. Se utiliza un enfoque basado en la arquitectura **encoder-decoder** con LSTM, entrenando el modelo con el dataset **ConvAI2** (Conversational Intelligence Challenge 2).

### Contenidos del práctico:

1. **Descarga del Dataset:**
   - Se utiliza el **dataset ConvAI2** del desafío Conversational Intelligence Challenge 2.
   - El dataset se descarga desde Google Drive utilizando el script `gdown`.

2. **Preprocesamiento de los Datos:**
   - Se limpia y tokeniza el conjunto de preguntas y respuestas usando un **tokenizador de Keras**.
   - Se crea un **vocabulario de palabras** con un tamaño máximo de 8000 palabras y se aplican secuencias de **padding** para que todas las preguntas y respuestas tengan la misma longitud.

3. **Carga de Embeddings:**
   - Se cargan los **embeddings de FastText** preentrenados para representar las palabras como vectores de **300 dimensiones**.
   - Estos embeddings mejoran la capacidad del modelo para entender el significado de las palabras, lo que resulta en un mejor desempeño durante el entrenamiento.

4. **Construcción del Modelo LSTM:**
   - El modelo sigue una arquitectura **encoder-decoder**, que utiliza **LSTM** en ambas partes:
     - **Encoder:** Procesa la secuencia de entrada (pregunta).
     - **Decoder:** Genera la secuencia de salida (respuesta) paso a paso.

5. **Entrenamiento del Modelo:**
   - El modelo se entrena con las preguntas y respuestas utilizando la **función de pérdida categorical_crossentropy** y el **optimizador Adam**.
   - Durante el entrenamiento, se visualizan gráficos de **exactitud** y **pérdida** para evaluar el desempeño del modelo.

6. **Inferencia:**
   - Tras el entrenamiento, se utiliza el modelo para generar respuestas a nuevas preguntas.
   - El proceso de inferencia incluye alimentar al **decoder** con la palabra generada en cada paso, y se decodifica la secuencia generada para obtener la respuesta.

7. **Evaluación de Resultados:**
   - Se presentan las respuestas generadas por el modelo para varias preguntas de ejemplo, como:
     - "Do you read?"
     - "Do you have any pet?"
     - "Where are you from?"

### Librerías utilizadas:
- **Python 3.x**
- **Keras** (para la construcción de las capas y el entrenamiento del modelo)
- **TensorFlow** (para la construcción y entrenamiento del modelo LSTM)
- **gdown** (para descargar el dataset desde Google Drive)
- **Numpy** (para manipulaciones numéricas)
- **Pandas** (para la manipulación y análisis de datos)
- **Matplotlib** y **Seaborn** (para la visualización de los resultados durante el entrenamiento)
- **scikit-learn** (para utilidades adicionales en el preprocesamiento y evaluación)

---

## Pasos generales para ejecutar los desafíos:

Para ejecutar los desafíos, sigue los siguientes pasos:

1. Abre los archivos correspondientes a cada desafío en Google Colab.
2. Asegúrate de tener todas las librerías requeridas instaladas.
3. Ejecuta las celdas en orden para completar cada uno de los pasos del desafío.
4. A medida que vayas ejecutando las celdas, revisa los resultados generados y los análisis que se realizan en cada paso.

