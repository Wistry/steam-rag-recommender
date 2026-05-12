# Sistema RAG para Recomendacion sobre Steam

Proyecto de la asignatura **Sistemas de Recuperacion y Recomendacion** del Master en Ciencia de Datos e Ingenieria de Computadores.

Este trabajo implementa un sistema de recomendacion basada en contenido para videojuegos de Steam usando una arquitectura **RAG** (*Retrieval-Augmented Generation*). El flujo combina embeddings semanticos, busqueda vectorial con **FAISS** y un **LLM local mediante Ollama** para recuperar juegos relevantes, reordenarlos y generar recomendaciones explicadas.

## Dataset

Se utiliza el dataset **Steam Video Game and Bundle Data**, en concreto:

- **Version 1: Review Data**
- **Version 1: User and Item Data**

Este conjunto de datos aporta informacion de apoyo para el sistema:

- reviews de usuarios
- interacciones usuario-item
- nombres de juegos
- textos usados para construir perfiles y documentos de juegos

## Referencias del dataset

- Wang-Cheng Kang, Julian McAuley. *Self-attentive sequential recommendation*. ICDM, 2018.
- Mengting Wan, Julian McAuley. *Item recommendation on monotonic behavior chains*. RecSys, 2018.
- Apurva Pathak, Kshitiz Gupta, Julian McAuley. *Generating and personalizing bundle recommendations on Steam*. SIGIR, 2017.

## Tecnologias principales

- Python
- Pandas
- NumPy
- Sentence Transformers
- FAISS
- Ollama

Modelos usados:

- Embeddings: `all-MiniLM-L6-v2`
- LLM: `llama3.2`

## Estructura general

```text
rag-steam-rag/
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_sesion1_indexacion.ipynb
│   ├── 02_sesion2_recuperacion.ipynb
│   └── 03_sesion3_rag_evaluacion.ipynb
├── outputs/
├── requirements.txt
└── README.md
```

## Flujo de trabajo

El proyecto esta dividido en tres sesiones encadenadas:

1. **Sesion 1: indexacion**
   - Lee los datos originales de Steam.
   - Construye documentos textuales por juego.
   - Genera embeddings con `all-MiniLM-L6-v2`.
   - Crea un indice FAISS para busqueda vectorial.

2. **Sesion 2: recuperacion**
   - Lee los artefactos generados en la sesion 1.
   - Construye perfiles de usuario.
   - Recupera candidatos top-k para cada usuario usando FAISS.
   - Guarda perfiles y resultados de recuperacion.

3. **Sesion 3: RAG y evaluacion**
   - Lee los candidatos recuperados en la sesion 2.
   - Construye prompts con el perfil del usuario y los candidatos.
   - Usa `llama3.2` mediante Ollama para reordenar y justificar recomendaciones.
   - Filtra alucinaciones comprobando que los titulos devueltos esten exactamente en la lista de candidatos.
   - Si el LLM devuelve menos de 3 recomendaciones validas, completa la salida con candidatos FAISS para que ningun usuario quede sin recomendaciones.

## Salidas generadas

Las sesiones guardan sus artefactos en subcarpetas separadas dentro de `outputs/`:

```text
outputs/
├── session1_indexing/
│   ├── games_dataset.csv
│   ├── embeddings.npy
│   └── faiss.index
├── session2_retrieval/
│   ├── user_profiles.csv
│   └── retrieval_results.csv
├── session3_rag/
│   ├── rag_outputs.json
│   ├── ranking_comparisons.csv
│   └── hallucinations.csv
```

Las carpetas `session1_indexing/`, `session2_retrieval/` y `session3_rag/` se crean automaticamente si no existen.

## Reproducibilidad

Los notebooks estan preparados para ejecutarse tanto desde la carpeta raiz del proyecto como desde `notebooks/`. Para ello detectan el directorio actual y construyen las rutas de forma relativa:

```python
PROJECT_ROOT = ".." if os.path.basename(os.getcwd()) == "notebooks" else "."
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")
```

Se usan rutas relativas (`..` o `.`) en lugar de rutas absolutas para evitar problemas de escritura con FAISS en Windows cuando la ruta completa contiene caracteres acentuados.

Para reproducir el pipeline completo, ejecutar los notebooks en este orden:

```text
01_sesion1_indexacion.ipynb
02_sesion2_recuperacion.ipynb
03_sesion3_rag_evaluacion.ipynb
```

La sesion 2 depende de los archivos de `outputs/session1_indexing/`. La sesion 3 depende de los archivos de `outputs/session1_indexing/` y `outputs/session2_retrieval/`.

## Entorno

Instalar dependencias:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para ejecutar la parte generativa de la sesion 3 hace falta tener Ollama disponible y el modelo `llama3.2` instalado:

```powershell
ollama run llama3.2
```

Si Ollama no esta disponible, la parte de recuperacion sigue siendo reproducible, pero la generacion de recomendaciones explicadas no podra ejecutarse.

## Validacion actual

Tras relanzar las tres sesiones con la estructura nueva se obtuvieron:

- `outputs/session1_indexing/games_dataset.csv`: 2058 juegos
- `outputs/session1_indexing/embeddings.npy`: 2058 embeddings de dimension 384
- `outputs/session2_retrieval/user_profiles.csv`: 20 perfiles de usuario
- `outputs/session2_retrieval/retrieval_results.csv`: 200 candidatos recuperados
- `outputs/session3_rag/rag_outputs.json`: 5 usuarios evaluados con 3 recomendaciones validas cada uno

La sesion 3 tambien guarda las alucinaciones detectadas en `hallucinations.csv`. Esto permite conservar evidencia de los casos en los que el LLM propuso titulos fuera de la lista de candidatos, aunque la salida final se complete con candidatos FAISS validos.
