# Sistema RAG para recomendacion de videojuegos de Steam

**Proyecto:** Sistemas de Recuperacion de Informacion y de Recomendacion  
**Tipo de sistema:** Recomendador basado en contenido con arquitectura RAG  
**Tecnologias principales:** Python, Pandas, NumPy, Sentence Transformers, FAISS y Ollama  

## Indice

1. Introduccion ........................................................................ 3  
2. Dataset Utilizado .................................................................. 4  
3. Arquitectura del proyecto ........................................................... 5  
4. Sesiones ........................................................................... 6  
   4.1. Sesion 1: procesado, embeddings e indexacion ............................. 6  
   4.2. Sesion 2: perfiles de usuario y recuperacion ............................. 6  
   4.3. Sesion 3: RAG, re-ranking y explicaciones ................................. 6  
5. Evaluacion ......................................................................... 7  
6. Conclusiones ....................................................................... 8  

## 1. Introduccion

Este proyecto implementa un sistema de recomendacion para videojuegos de Steam basado en recuperacion semantica y generacion aumentada por recuperacion, o RAG (*Retrieval-Augmented Generation*). El objetivo principal es construir un flujo completo que, partiendo de reseñas e interacciones de usuarios, sea capaz de generar recomendaciones personalizadas y justificadas.

El sistema combina dos ideas complementarias. Por un lado, utiliza embeddings semanticos para representar juegos y perfiles de usuario en un mismo espacio vectorial. Esto permite recuperar candidatos similares mediante busqueda vectorial con FAISS. Por otro lado, incorpora un modelo de lenguaje local, ejecutado con Ollama, para reordenar los candidatos recuperados y generar explicaciones breves adaptadas al perfil del usuario.

El proyecto esta organizado en tres sesiones encadenadas:

- La primera sesion procesa el dataset, construye documentos por juego, genera embeddings y crea el indice FAISS.
- La segunda sesion construye perfiles de usuario y recupera juegos candidatos para cada perfil.
- La tercera sesion aplica el componente RAG, usando los candidatos de FAISS como contexto para el LLM, y añade mecanismos de evaluacion y control de alucinaciones.

El resultado final no es solo una lista de recomendaciones, sino una salida mas interpretable: cada usuario recibe juegos recomendados junto con una explicacion del motivo por el que encajan con sus preferencias. Esta separacion entre recuperacion y generacion tambien mejora el control del sistema, porque el LLM no recomienda libremente desde cero, sino sobre una lista previa de candidatos validos.

## 2. Dataset Utilizado

El proyecto utiliza el dataset **Steam Video Game and Bundle Data**, asociado a trabajos de investigacion sobre recomendacion de items, comportamiento de usuarios y bundles en Steam. En concreto, se emplean datos de reseñas, usuarios e items para construir tanto la coleccion de juegos como los perfiles de usuario.

Las referencias principales del dataset son:

- Wang-Cheng Kang y Julian McAuley, *Self-attentive sequential recommendation*, ICDM 2018.
- Mengting Wan y Julian McAuley, *Item recommendation on monotonic behavior chains*, RecSys 2018.
- Apurva Pathak, Kshitiz Gupta y Julian McAuley, *Generating and personalizing bundle recommendations on Steam*, SIGIR 2017.

El dataset aporta informacion de varios tipos:

- Reseñas de usuarios sobre videojuegos.
- Identificadores de juegos e identificadores de usuario.
- Nombres de juegos.
- Interacciones usuario-item.
- Texto suficiente para construir documentos representativos de cada juego.

Durante el procesado se genera un dataset intermedio de juegos en `outputs/session1_indexing/games_dataset.csv`. Este archivo contiene 2.058 juegos y las columnas:

| Campo | Descripcion |
| --- | --- |
| `item_id` | Identificador del juego en Steam. |
| `item_name` | Nombre real del juego. |
| `num_reviews` | Numero de reseñas usadas para construir el documento. |
| `document` | Texto agregado y limpiado que representa el contenido del juego. |

Cada documento combina el nombre del juego con reseñas positivas limpias. De esta forma, el embedding de cada juego no depende solo del titulo, sino tambien del lenguaje real usado por los usuarios para describir su experiencia: genero, mecanicas, calidad percibida, modo de juego, dificultad o comparaciones con otros titulos.

## 3. Arquitectura del proyecto

La arquitectura del proyecto sigue un pipeline RAG dividido en artefactos reutilizables. Cada sesion produce salidas que sirven como entrada para la siguiente, lo que facilita reproducir, inspeccionar y depurar el sistema por partes.

La estructura principal es:

```text
rag-steam-rag/
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_sesion1_indexacion.ipynb
│   ├── 02_sesion2_recuperacion.ipynb
│   └── 03_sesion3_rag_evaluacion.ipynb
├── outputs/
│   ├── session1_indexing/
│   ├── session2_retrieval/
│   └── session3_rag/
├── requirements.txt
└── README.md
```

Las dependencias principales son:

- `pandas` y `numpy` para carga, limpieza y transformacion de datos.
- `sentence-transformers` para generar embeddings semanticos.
- `faiss-cpu` para indexacion y busqueda vectorial.
- `ollama` para llamar a un LLM local.

El modelo de embeddings usado es `all-MiniLM-L6-v2`, que genera vectores de dimension 384. Los embeddings se normalizan con norma L2 y se indexan en FAISS mediante `IndexFlatIP`. Al estar normalizados, el producto interno equivale a similitud coseno, una medida adecuada para comparar representaciones semanticas.

El componente generativo usa `llama3.2` mediante Ollama. Su papel no es buscar en todo el dataset, sino recibir un conjunto cerrado de candidatos y decidir cuales son las mejores recomendaciones finales para un usuario. Esta decision se acompaña de una justificacion en lenguaje natural.

Los artefactos generados quedan separados por sesion:

| Carpeta | Archivos principales | Funcion |
| --- | --- | --- |
| `outputs/session1_indexing/` | `games_dataset.csv`, `embeddings.npy`, `faiss.index` | Coleccion vectorizada e indice de busqueda. |
| `outputs/session2_retrieval/` | `user_profiles.csv`, `retrieval_results.csv` | Perfiles de usuario y candidatos recuperados. |
| `outputs/session3_rag/` | `rag_outputs.json`, `ranking_comparisons.csv`, `hallucinations.csv` | Respuestas RAG, comparacion de ranking y alucinaciones detectadas. |

## 4. Sesiones

### 4.1. Sesion 1: procesado, embeddings e indexacion

La primera sesion construye la base del sistema. Como las reseñas no siempre contienen directamente el nombre del juego, primero se genera un mapeo entre `item_id` e `item_name` a partir de los datos de usuarios e items. Despues se leen las reseñas, se limpian los textos y se agregan reseñas positivas por juego.

El resultado de esta fase es un documento textual por item. Cada documento esta formado por el nombre del juego y un conjunto de reseñas limpias. Esta decision permite que el sistema capture informacion semantica mas rica que la disponible en el titulo: por ejemplo si un juego se percibe como cooperativo, competitivo, de supervivencia, de estrategia o similar a otros juegos populares.

Una vez construidos los documentos, se generan embeddings con `all-MiniLM-L6-v2`. Estos vectores se normalizan y se guardan en `embeddings.npy`. A continuacion se crea un indice FAISS `IndexFlatIP`, que se guarda como `faiss.index`.

Salidas principales de la sesion:

- `games_dataset.csv`: 2.058 juegos procesados.
- `embeddings.npy`: 2.058 embeddings de dimension 384.
- `faiss.index`: indice vectorial para busqueda por similitud.

La sesion incluye tambien una prueba rapida de busqueda para comprobar que, dada una consulta textual, el indice devuelve juegos semanticamente relacionados.

### 4.2. Sesion 2: perfiles de usuario y recuperacion

La segunda sesion utiliza los artefactos de la sesion anterior para construir recomendaciones candidatas. Primero se cargan el dataset de juegos, los embeddings y el indice FAISS. Despues se construyen perfiles de usuario a partir de los juegos consumidos por cada usuario.

Cada perfil textual resume los juegos asociados a un usuario y se transforma en un embedding usando el mismo modelo que en la sesion 1. Al compartir espacio vectorial con los juegos, el perfil puede usarse como consulta semantica contra FAISS.

El sistema recupera un ranking top-k de juegos candidatos para cada usuario. En la ejecucion actual se generaron:

- `user_profiles.csv`: 20 perfiles de usuario.
- `retrieval_results.csv`: 200 candidatos recuperados, equivalentes a 10 candidatos por usuario.

El archivo `retrieval_results.csv` conserva informacion util para la siguiente etapa: `user_id`, posicion del ranking, `item_id`, `item_name`, puntuacion de similitud, texto del perfil y nombres de juegos consumidos. Esta salida es la entrada principal del sistema RAG.

### 4.3. Sesion 3: RAG, re-ranking y explicaciones

La tercera sesion completa el pipeline. A partir de los candidatos recuperados por FAISS, se construye un prompt para cada usuario. El prompt incluye el perfil del usuario y una lista limitada de candidatos validos. El LLM debe seleccionar las mejores recomendaciones, reordenarlas si lo considera necesario y explicar brevemente la razon de cada seleccion.

El modelo usado es `llama3.2` mediante Ollama. Para hacer la salida mas controlable, se solicita una respuesta estructurada en JSON. Despues, el notebook extrae el JSON de la respuesta, valida los titulos y aplica un filtro de alucinaciones.

El filtro de alucinaciones comprueba que cada titulo recomendado por el LLM exista exactamente entre los candidatos recuperados para ese usuario. Si el modelo propone un juego que no estaba en la lista, el elemento se registra en `hallucinations.csv` y no se considera recomendacion valida. Ademas, si tras el filtrado quedan menos de 3 recomendaciones, el sistema completa la salida con candidatos FAISS para garantizar una respuesta final util.

Tambien se genera una comparacion entre el ranking original de FAISS y el ranking final propuesto por el LLM. Esta comparacion se almacena en `ranking_comparisons.csv` y permite observar si el LLM actua solo como explicador o si modifica realmente el orden de los candidatos.

## 5. Evaluacion

La evaluacion del proyecto se plantea desde tres perspectivas: calidad de recuperacion, control de la generacion y utilidad de las explicaciones.

En recuperacion, la ejecucion actual produjo 200 candidatos para 20 usuarios. Las puntuaciones de similitud de FAISS se movieron aproximadamente entre 0,3465 y 0,7271, con una media de 0,6139. Esto indica que, para la mayoria de usuarios, el sistema encuentra juegos semanticamente cercanos a sus perfiles.

En la parte RAG se evaluaron 5 usuarios de demostracion. El sistema genero 15 recomendaciones validas finales, es decir, 3 por usuario. Se detectaron 3 alucinaciones en total y 4 de las 5 respuestas no tuvieron ninguna alucinacion. La existencia del archivo `hallucinations.csv` es importante porque deja evidencia de los casos en los que el LLM intento recomendar elementos fuera de la lista de candidatos.

Resumen de resultados:

| Metrica | Valor |
| --- | ---: |
| Juegos procesados | 2.058 |
| Dimension de embeddings | 384 |
| Perfiles de usuario | 20 |
| Candidatos recuperados | 200 |
| Usuarios evaluados con RAG | 5 |
| Recomendaciones validas finales | 15 |
| Alucinaciones detectadas | 3 |
| Respuestas sin alucinaciones | 4 de 5 |

La comparacion de ranking muestra 15 elementos evaluados, con cambios entre -2 y +3 posiciones y un cambio medio de 0,2. Esto sugiere que el LLM no altera de forma extrema el ranking de FAISS, pero si puede ajustar el orden final cuando las explicaciones o el perfil del usuario lo justifican.

La evaluacion cualitativa se centra en comprobar que las recomendaciones:

- Pertenecen a los candidatos recuperados.
- Son coherentes con el historial o perfil textual del usuario.
- Incluyen justificaciones especificas y no genericas.
- Mantienen una salida final completa incluso cuando el LLM produce elementos no validos.

Esta combinacion de controles hace que el sistema sea mas robusto que una generacion libre, porque la recuperacion restringe el espacio de respuesta y el filtro posterior elimina recomendaciones no verificables.

## 6. Conclusiones

El proyecto construye un sistema RAG funcional para recomendacion de videojuegos de Steam. La solucion parte de datos reales de reseñas e interacciones, transforma los juegos en documentos textuales, genera embeddings semanticos, indexa la coleccion con FAISS y usa perfiles de usuario como consultas de recuperacion.

La division en tres sesiones facilita entender y reproducir el trabajo. La sesion 1 deja preparada la base vectorial; la sesion 2 convierte el historial de usuario en recuperacion personalizada; y la sesion 3 incorpora un LLM local para re-ranking, explicaciones y validacion de respuestas.

Uno de los puntos mas relevantes del proyecto es el control de alucinaciones. El LLM aporta lenguaje natural y capacidad de razonamiento sobre candidatos, pero no se le permite recomendar cualquier juego. Sus respuestas se contrastan con la lista recuperada por FAISS, y los casos invalidos se registran para analisis posterior.

Como posibles mejoras futuras se podrian incorporar metricas clasicas de recomendacion, como precision@k, recall@k o NDCG, siempre que se disponga de un conjunto de validacion con interacciones relevantes. Tambien seria interesante comparar distintos modelos de embeddings, aumentar el numero de usuarios evaluados con RAG y probar estrategias de prompt mas estrictas para reducir todavia mas las alucinaciones.

En conjunto, el sistema demuestra un flujo completo y modular de recomendacion semantica explicable: recupera candidatos con eficiencia, genera recomendaciones personalizadas y mantiene trazabilidad sobre los errores del componente generativo.
