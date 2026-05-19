# VeriFact-AI Pro: Detector Inteligente Local

Proyecto desarrollado para la asignatura de **Programación de Inteligencia Artificial**.

## Características
* **Análisis 100% Local:** Implementa un modelo predictivo offline que no depende de APIs externas ni sufre bloqueos de IP.
* **Procesamiento del Lenguaje Natural:** Extracción de características lingüísticas mediante **TF-IDF**.
* **Modelo Estadístico:** Clasificador matemático basado en el algoritmo **Naive Bayes (MultinomialNB)** entrenado con un dataset estratégico en español.
* **Interfaz Gráfica Avanzada:** Desarrollado con **Streamlit** y extracción de artículos web mediante **Newspaper3K**.

## Instalación y Uso
1. Instalar dependencias: `pip install streamlit scikit-learn newspaper3k`
2. Ejecutar la aplicación: `python -m streamlit run src/app.py`
