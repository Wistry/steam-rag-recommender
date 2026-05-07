# Sistema RAG para Recomendacion sobre Steam

Proyecto de la asignatura **Sistemas de Recuperacion y Recomendacion** del Master en Ciencia de Datos e Ingenieria de Computadores.

Este trabajo implementa un sistema de recomendacion basada en contenido para videojuegos de Steam usando una arquitectura **RAG** (*Retrieval-Augmented Generation*). El flujo general combina embeddings, busqueda vectorial con **FAISS** y un **LLM** para recuperar juegos relevantes y generar recomendaciones explicadas.

## Dataset

Se utiliza el dataset **Steam Video Game and Bundle Data**, en concreto:

- **Version 1: Review Data**
- **Version 1: User and Item Data**

Este conjunto de datos aporta informacion de apoyo para el sistema, como:

- reviews de usuarios
- metadatos de juegos
- interacciones usuario-item
- generos y otra informacion descriptiva

## Referencias del dataset

- Wang-Cheng Kang, Julian McAuley. *Self-attentive sequential recommendation*. ICDM, 2018.
- Mengting Wan, Julian McAuley. *Item recommendation on monotonic behavior chains*. RecSys, 2018.
- Apurva Pathak, Kshitiz Gupta, Julian McAuley. *Generating and personalizing bundle recommendations on Steam*. SIGIR, 2017.

## Tecnologias principales

- Python
- Sentence Transformers
- FAISS
- Pandas
- NumPy
- Ollama

Modelos usados:

- Embeddings: `all-MiniLM-L6-v2`
- LLM: `llama3.2`

## Estructura general

```text
rag-steam-rag/
├── data/
├── notebooks/
├── src/
├── outputs/
├── informe/
├── requirements.txt
└── README.md
```
