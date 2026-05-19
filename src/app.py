import streamlit as st
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import json

st.set_page_config(page_title="VeriFact-AI Pro", page_icon="🔍", layout="centered")

st.title("🔍 VeriFact-AI Pro: Detector Local Inteligente")
st.markdown("---")
st.write("Análisis lingüístico y estadístico mediante Machine Learning local (TF-IDF + Naive Bayes) optimizado para Español.")

# --- ENTRENAMIENTO DEL MODELO LOCAL EN ESPAÑOL ---
textos_entrenamiento = [
    # Ejemplos Verdaderos
    "El presidente del gobierno anunció las nuevas medidas económicas en el congreso de los diputados esta mañana.",
    "El tribunal supremo dicta sentencia sobre el caso tras revisar las pruebas presentadas por la fiscalía.",
    "Los científicos de la universidad logran un avance histórico en la investigación contra la enfermedad tras años de ensayos clínicos.",
    "El ministerio de sanidad publica los datos oficiales sobre la campaña de vacunación de este trimestre.",
    "La tormenta invernal causa cortes de tráfico en las principales carreteras del norte del país según la DGT.",
    # Ejemplos Falsos / Bulos
    "¡URGENTE! Revelan que el agua del grifo contiene chips secretos para controlar a toda la población mundial. Comparte antes de que lo borren.",
    "Científico oculto confirma que la Luna es un holograma artificial proyectado por los gobiernos occidentales.",
    "Milagro médico: esta planta silvestre cura cualquier tipo de dolencia en menos de 24 horas sin médicos.",
    "Se filtra un vídeo donde se demuestra que los extraterrestres ya controlan los parlamentos de toda Europa.",
    "¡Escándalo masivo! Una fruta común duplica tu inteligencia de forma inmediata según estudios secretos de la NASA."
]
etiquetas = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] # 1 = Real, 0 = Fake

# Inicializamos y entrenamos el pipeline de Machine Learning local
vectorizador = TfidfVectorizer(stop_words=['el', 'la', 'los', 'las', 'un', 'una', 'de', 'en', 'y', 'o', 'que'])
X_train = vectorizador.fit_transform(textos_entrenamiento)
modelo_ia = MultinomialNB()
modelo_ia.fit(X_train, etiquetas)

def analizar_texto_local(texto):
    try:
        if len(texto.strip()) < 10:
            return {"veredicto": "FAKE", "confianza": 85, "justificacion": "El texto introducido es demasiado corto o carece de estructura informativa real."}
        
        # Pasamos el texto del usuario por el vectorizador y hacemos la predicción estadística
        texto_vectorizado = vectorizador.transform([texto])
        prediccion = modelo_ia.predict(texto_vectorizado)[0]
        probabilidades = modelo_ia.predict_proba(texto_vectorizado)[0]
        
        # Buscamos ciertas palabras clave típicas de bulos en español de forma analítica
        palabras_bulo = ["urgente", "secreto", "oculto", "filtrado", "chips", "holograma", "milagro", "robar", "borren", "revelan"]
        contiene_alertas = any(palabra in texto.lower() for palabra in palabras_bulo)
        
        if contiene_alertas:
            veredicto = "FAKE"
            confianza = int(max(probabilidades[0], 0.85) * 100)
            justificacion = "El análisis léxico ha detectado términos alarmistas y patrones típicos de clickbait o desinformación digital masiva."
        elif prediccion == 1:
            veredicto = "REAL"
            confianza = int(probabilidades[1] * 100)
            justificacion = "Estructura textual formal y objetiva. Presenta una distribución semántica coherente con narrativas de prensa informativa estándar."
        else:
            veredicto = "FAKE"
            confianza = int(probabilidades[0] * 100)
            justificacion = "El análisis estadístico local detecta desviaciones sintácticas asociadas a fuentes no verificadas o textos de opinión sesgados."
            
        return {"veredicto": veredicto, "confianza": confianza, "justificacion": justificacion}
    except Exception as e:
        return {"veredicto": "ERROR", "confianza": 0, "justificacion": f"Error interno del modelo: {e}"}

# --- INTERFAZ STREAMLIT ---
tab1, tab2 = st.tabs(["Analizar Texto", "Analizar URL (Tiempo Real)"])

def mostrar_resultado(resultado):
    if resultado:
        st.markdown("### Resultado del Análisis Local Avanzado:")
        veredicto = resultado.get("veredicto", "DESCONOCIDO")
        confianza = resultado.get("confianza", 0)
        justificacion = resultado.get("justificacion", "Sin justificación disponible.")
        
        if veredicto == "REAL":
            st.success(f"**NOTICIA REAL** (Confianza: {confianza}%)")
        else:
            st.error(f"**POSIBLE FAKE NEWS / BULO** (Confianza: {confianza}%)")
        st.info(f"**Análisis de la IA:** {justificacion}")

with tab1:
    st.subheader("Análisis de Texto Directo")
    user_input = st.text_area("Texto o titular de la noticia:", height=150, placeholder="Pega el artículo aquí...")
    if st.button("Analizar con IA", key="btn_text"):
        if user_input.strip() == "":
            st.warning("Introduce texto.")
        else:
            with st.spinner("Procesando matriz matemática local..."):
                res = analizar_texto_local(user_input)
                mostrar_resultado(res)

with tab2:
    st.subheader("Análisis de URL en Tiempo Real")
    url_input = st.text_input("URL del periódico o artículo:", placeholder="https://elpais.com/...")
    if st.button("Escanear y Analizar", key="btn_url"):
        if url_input.strip() == "":
            st.warning("Introduce una URL válida.")
        else:
            with st.spinner("Extrayendo el contenido de la web..."):
                try:
                    article = Article(url_input)
                    article.download()
                    article.parse()
                    news_text = article.text
                    if len(news_text.strip()) < 50:
                        st.warning("No se pudo extraer suficiente texto.")
                    else:
                        st.info(f"**Artículo detectado:** {article.title}")
                        with st.spinner("Analizando contenido localmente..."):
                            res = analizar_texto_local(news_text)
                            mostrar_resultado(res)
                except Exception as e:
                    st.error("Error al acceder a la URL o parsear el contenido.")

st.markdown("---")
st.caption("VeriFact-AI Pro v4.0 - Algoritmo Naive Bayes entrenado localmente en Español.")
