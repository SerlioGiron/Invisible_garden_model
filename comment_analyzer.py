import os
import sys
import asyncio
import json
import re
from datetime import datetime

# Configuración para Windows y PyTorch
if sys.platform == "win32":
    # 1. Deshabilitar completamente el file watcher
    os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"
    
    # 2. Configurar el event loop policy
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 3. Parchear el sistema de clases de Torch
    try:
        import torch
        torch.classes.__path__ = []  # Elimina la inspección de rutas conflictivas
    except ImportError:
        pass  # Torch no está instalado

import streamlit as st    
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# Configurar claves API
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("No se encontró la clave API de Groq. Asegúrate de configurar GROQ_API_KEY en tu archivo .env")

model = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")

st.set_page_config(page_title="Analizador de Comentarios", layout="wide")

@st.cache_data
def load_tags():
    """Cargar tags desde el archivo tags.txt"""
    try:
        with open("tags.txt", "r", encoding="utf-8") as file:
            tags = [line.strip().lower() for line in file if line.strip()]
        return tags
    except FileNotFoundError:
        st.error("No se encontró el archivo tags.txt")
        return []
    except Exception as e:
        st.error(f"Error cargando tags: {str(e)}")
        return []

def is_coherent_text(text):
    """Validar si el texto es coherente (más de una palabra y tiene sentido básico)"""
    text = text.strip()
    
    # 1. Verificar que no esté vacío
    if not text:
        return False
    
    # 2. Verificar que tenga al menos 2 palabras
    words = text.split()
    if len(words) < 2:
        return False
    
    # 3. Verificar que no sean solo caracteres especiales o números
    clean_text = re.sub(r'[^\w\s]', '', text)
    if len(clean_text.strip()) < 3:
        return False
    
    # 4. Verificar que no sean solo repeticiones de la misma palabra
    unique_words = set(word.lower() for word in words if word.isalpha())
    if len(unique_words) < 2:
        return False
    
    # 5. Detectar incoherencia semántica usando reglas híbridas
    
    # Lista de patrones específicamente incoherentes
    incoherent_patterns = [
        "casa azul mojado", "perro volando matemáticas", "mesa correr feliz",
        "computadora cantar verde", "silla bailar número", "árbol escribir calor",
        "teléfono dormir azúcar", "libro nadar rojo", "ventana comer fríos",
        "zapato volar música", "reloj bailar agua", "puerta correr números"
    ]
    
    # Si es exactamente uno de estos casos, es incoherente
    if text.lower().strip() in incoherent_patterns:
        return False
    
    # Verificar patrones de incoherencia (sustantivo + verbo incongruente + adjetivo/sustantivo)
    # Ejemplo: "casa cantar azul" (objeto físico + acción incompatible + descriptor)
    if len(words) == 3:
        # Objetos físicos que no pueden realizar ciertas acciones
        objects = {'casa', 'mesa', 'silla', 'puerta', 'ventana', 'libro', 'teléfono', 'computadora'}
        impossible_actions = {'cantar', 'bailar', 'correr', 'volar', 'nadar', 'dormir', 'comer'}
        
        word1, word2, word3 = [w.lower() for w in words]
        
        # Si primer palabra es objeto y segunda es acción imposible
        if word1 in objects and word2 in impossible_actions:
            return False
    
    # Verificar estructura mínima de oración en español
    structure_indicators = {
        # Artículos
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        # Verbos auxiliares y comunes
        'es', 'está', 'son', 'están', 'tiene', 'tienen', 'hay', 'fue', 'era',
        # Preposiciones
        'de', 'del', 'en', 'con', 'por', 'para', 'desde', 'hasta', 'sobre',
        # Pronombres y conectores
        'que', 'se', 'me', 'te', 'le', 'nos', 'les', 'mi', 'tu', 'su',
        # Adverbios y conjunciones
        'muy', 'más', 'menos', 'bien', 'mal', 'no', 'sí', 'y', 'o', 'pero',
        # Verbos de opinión/estado
        'quiero', 'necesito', 'creo', 'pienso', 'siento', 'veo', 'escucho',
        'necesitamos', 'queremos', 'podemos', 'debemos'
    }
    
    text_words = set(word.lower() for word in words)
    has_structure = bool(text_words.intersection(structure_indicators))
    
    # Si no tiene indicadores estructurales, probablemente es incoherente
    if not has_structure:
        return False
    
    return True

def extract_tags_from_text(text, available_tags):
    """Extraer tags relevantes del texto - cantidad dinámica según longitud"""
    text_lower = text.lower()
    tag_scores = {}  # tag -> puntuación de relevancia
    
    # Palabras del texto
    words_in_text = re.findall(r'\b\w+\b', text_lower)
    
    # Determinar cantidad máxima de tags según longitud del texto
    if len(words_in_text) <= 4:  # Texto muy corto (4 palabras o menos)
        max_tags = 1
    elif len(words_in_text) <= 8:  # Texto mediano (5-8 palabras)
        max_tags = 2
    else:  # Texto largo (9+ palabras)
        max_tags = 3
    
    for tag in available_tags:
        score = 0
        matches = []  # Posiciones donde se encuentra el tag
        
        # 1. Búsqueda exacta
        for i, word in enumerate(words_in_text):
            if word == tag:
                matches.append(i)
                score += 3  # Puntuación alta para coincidencia exacta
        
        # 2. Búsqueda de variaciones simples
        if not matches:  # Solo si no hubo coincidencia exacta
            for i, word in enumerate(words_in_text):
                # Plurales simples: maestro -> maestros
                if word == tag + 's' or tag == word + 's':
                    matches.append(i)
                    score += 2  # Puntuación media para variaciones
                    break
                # Plurales con 'es': clase -> clases
                elif word == tag + 'es' or tag == word + 'es':
                    matches.append(i)
                    score += 2
                    break
        
        # Si encontramos el tag, calcular puntuación final
        if matches:
            # Bonus por longitud del tag (tags más específicos son más relevantes)
            score += len(tag) * 0.1
            
            # Bonus por posición temprana en el texto
            first_position = min(matches)
            position_bonus = max(0, 5 - first_position)  # Más puntos si aparece al principio
            score += position_bonus
            
            # Bonus por frecuencia
            frequency_bonus = len(matches) * 0.5
            score += frequency_bonus
            
            tag_scores[tag] = score
    
    # Ordenar por relevancia (mayor puntuación primero) y tomar según longitud del texto
    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    most_relevant_tags = [tag for tag, score in sorted_tags[:max_tags]]
    
    return most_relevant_tags

def categorize_comment(comment):
    """Categorizar el comentario usando LLM - Soporta español e inglés"""
    template = """
    Analyze the following comment and categorize it EXACTLY into one of these four categories:
    - "Sugerencia": If the comment proposes improvements, ideas, changes or constructive recommendations
    - "Opinion": If the comment expresses a neutral or positive personal opinion, experiences without being offensive
    - "HateSpeech": If the comment contains offensive language, discrimination, threats, insults, very negative criticism, or very negative feelings towards people (e.g.: "the teacher is bad", "I hate...", "it's terrible", etc.)
    - "Vida universitaria": If the comment specifically refers to experiences, situations, activities or aspects of university, academic or student life that do not fit into the other categories

    Important rules:
    1. Respond ONLY with one of these four words: "Sugerencia", "Opinion", "HateSpeech", or "Vida universitaria"
    2. Do not add additional explanations
    3. Negative comments about people (teachers, classmates, etc.) go in "HateSpeech"
    4. Comments about classes, university, studies, campus, etc. go in "Vida universitaria"
    5. If in doubt, prioritize in this order: HateSpeech > Vida universitaria > Sugerencia > Opinion
    6. THE COMMENT CAN BE IN SPANISH OR ENGLISH - Detect the language and analyze accordingly

    Examples (Spanish):
    - "El maestro es malo" → HateSpeech
    - "La clase de matemáticas es difícil" → Vida universitaria
    - "Deberían mejorar la cafetería" → Sugerencia
    - "Me gusta estudiar" → Opinion
    
    Examples (English):
    - "The teacher is terrible" → HateSpeech
    - "Math class is challenging" → Vida universitaria
    - "They should improve the cafeteria" → Sugerencia
    - "I enjoy studying" → Opinion

    Comment: "{comment}"
    
    Category:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    try:
        response = ""
        for chunk in chain.stream({"comment": comment}):
            response += chunk.content
        
        # Limpiar la respuesta y validar
        category = response.strip()
        valid_categories = ["Sugerencia", "Opinion", "HateSpeech", "Vida universitaria"]
        
        if category in valid_categories:
            return category
        else:
            # Si la respuesta no es válida, intentar extraer una categoría válida
            for valid_cat in valid_categories:
                if valid_cat.lower() in category.lower():
                    return valid_cat
            # Si no se encuentra ninguna, retornar Opinion por defecto
            return "Opinion"
            
    except Exception as e:
        st.error(f"Error categorizando comentario: {str(e)}")
        return "Opinion"  # Categoría por defecto en caso de error

def formalize_hate_speech(comment):
    """Convertir comentario ofensivo a lenguaje formal y apropiado - Soporta español e inglés"""
    template = """
    You are a language moderator. The following comment contains offensive language. Convert it to a formal, respectful and constructive comment that expresses the same idea but in a manner appropriate for an academic or professional environment.

    CRITICAL RULES:
    1. Remove all offensive words, vulgarities or insults
    2. Maintain the essence of the message but in a constructive tone
    3. Use formal and respectful language
    4. If it's a complaint, convert it to constructive feedback
    5. Maximum 2-3 lines
    6. Respond ONLY with the formalized text, without additional explanations
    7. **MANDATORY**: If the original comment is in ENGLISH, respond in ENGLISH. If it's in SPANISH, respond in SPANISH. NEVER change the language.

    Examples:
    - Original (English): "This teacher sucks" → Formalized (English): "I believe the teaching methodology could be improved"
    - Original (Spanish): "Este profesor es horrible" → Formalized (Spanish): "Considero que la metodología de enseñanza podría mejorar"

    Original comment: "{comment}"

    Formalized comment (in the SAME language as the original):
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    try:
        response = ""
        for chunk in chain.stream({"comment": comment}):
            response += chunk.content
        
        # Limpiar la respuesta
        formalized = response.strip()
        
        # Quitar comillas si las hay
        if formalized.startswith('"') and formalized.endswith('"'):
            formalized = formalized[1:-1]
        elif formalized.startswith("'") and formalized.endswith("'"):
            formalized = formalized[1:-1]
        
        # Quitar cualquier comilla doble o simple sobrante
        formalized = formalized.replace('""', '').replace("''", '')
        
        # Si la respuesta está vacía o muy corta, usar una versión genérica
        if len(formalized) < 10:
            formalized = "Comentario convertido a lenguaje apropiado por contener contenido ofensivo."
        
        return formalized
        
    except Exception as e:
        st.error(f"Error formalizando comentario: {str(e)}")
        return "Comentario modificado por contener contenido inapropiado."

def save_to_json(data, filename="comentarios_analizados.json"):
    """Guardar datos en archivo JSON"""
    try:
        # Cargar datos existentes si el archivo existe
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
        else:
            existing_data = []
        
        # Agregar nuevo análisis
        existing_data.append(data)
        
        # Guardar datos actualizados
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Error guardando en JSON: {str(e)}")
        return False

def load_analysis_history():
    """Cargar historial de análisis desde JSON"""
    try:
        if os.path.exists("comentarios_analizados.json"):
            with open("comentarios_analizados.json", "r", encoding="utf-8") as file:
                return json.load(file)
        return []
    except Exception as e:
        st.error(f"Error cargando historial: {str(e)}")
        return []

def display_statistics(history):
    """Mostrar estadísticas del análisis"""
    if not history:
        return
    
    # Contar categorías
    categories = [item["categoria"] for item in history]
    category_counts = {
        "Sugerencia": categories.count("Sugerencia"),
        "Opinion": categories.count("Opinion"),
        "HateSpeech": categories.count("HateSpeech"),
        "Vida universitaria": categories.count("Vida universitaria")
    }
    
    # Mostrar estadísticas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sugerencias", category_counts["Sugerencia"])
    with col2:
        st.metric("Opiniones", category_counts["Opinion"])
    with col3:
        st.metric("Hate Speech", category_counts["HateSpeech"])
    with col4:
        st.metric("Vida Universitaria", category_counts["Vida universitaria"])
    
    # Tags más comunes
    all_tags = []
    for item in history:
        all_tags.extend(item["tags"])
    
    if all_tags:
        from collections import Counter
        tag_counts = Counter(all_tags)
        st.subheader("Tags más comunes:")
        for tag, count in tag_counts.most_common(10):
            st.write(f"• {tag}: {count} veces")

def main():
    st.title("🎓 Analizador de Comentarios Universitarios")
    st.markdown("### Detecta automáticamente si un comentario es una Sugerencia, Opinión, Hate Speech o sobre Vida Universitaria")
    
    # Cargar tags disponibles
    available_tags = load_tags()
    if not available_tags:
        st.warning("No se pudieron cargar los tags. El análisis continuará sin detección de tags.")
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("📊 Configuración")
        
        # Mostrar estadísticas
        history = load_analysis_history()
        st.subheader(f"Análisis realizados: {len(history)}")
        
        if history:
            display_statistics(history)
        
        # Opción para descargar historial
        if st.button("📥 Descargar Historial JSON"):
            if history:
                json_str = json.dumps(history, ensure_ascii=False, indent=2)
                st.download_button(
                    label="Descargar comentarios_analizados.json",
                    data=json_str,
                    file_name="comentarios_analizados.json",
                    mime="application/json"
                )
        
        # Opción para limpiar historial
        if st.button("🗑️ Limpiar Historial"):
            if os.path.exists("comentarios_analizados.json"):
                os.remove("comentarios_analizados.json")
                st.success("Historial limpiado")
                st.rerun()
    
    # Input principal
    st.subheader("💬 Ingresa un comentario para analizar:")
    
    # Inicializar session state para el comentario y el estado de formalización
    if "comment_text" not in st.session_state:
        st.session_state.comment_text = ""
    if "show_formalized_message" not in st.session_state:
        st.session_state.show_formalized_message = False
    if "last_comment" not in st.session_state:
        st.session_state.last_comment = ""
    if "is_formalized_comment" not in st.session_state:
        st.session_state.is_formalized_comment = False
    
    # Mostrar mensaje si el comentario fue formalizado
    if st.session_state.show_formalized_message:
        st.warning("⚠️ **Versión formalizada ya que se consideró ofensiva su mensaje anterior:**")
    
    comment_input = st.text_area(
        "Comentario:",
        value=st.session_state.comment_text,
        placeholder="Escribe aquí el comentario que quieres analizar...",
        height=100,
        key="comment_input"
    )
    
    # Si el usuario cambió el texto, ocultar el mensaje de formalización
    if comment_input != st.session_state.last_comment and st.session_state.show_formalized_message:
        if comment_input != st.session_state.comment_text:  # Solo si realmente editó
            st.session_state.show_formalized_message = False
    
    st.session_state.last_comment = comment_input
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        analyze_button = st.button("🔍 Analizar Comentario", type="primary")
    
    if analyze_button and comment_input.strip():
        # Validar que el texto sea coherente
        if not is_coherent_text(comment_input):
            st.error("⚠️ Se tiene que escribir algo coherente")
        else:
            with st.spinner("Analizando comentario..."):
                # 1. Categorizar comentario original
                original_category = categorize_comment(comment_input)
                
                # 2. Si es HateSpeech y NO ha sido formalizado aún, formalizarlo
                if original_category == "HateSpeech" and not st.session_state.is_formalized_comment:
                    final_comment = formalize_hate_speech(comment_input)
                    # Actualizar el texto en el session state para que se refleje en el input
                    st.session_state.comment_text = final_comment
                    st.session_state.show_formalized_message = True
                    st.session_state.is_formalized_comment = True
                    st.rerun()  # Recargar para mostrar el texto actualizado
                
                # 3. Si llegamos aquí, procesar normalmente (incluso si es HateSpeech formalizado)
                final_comment = comment_input
                final_category = original_category
                
                # Si es un comentario que fue formalizado, mantener la categoría como HateSpeech
                if st.session_state.is_formalized_comment and st.session_state.show_formalized_message:
                    final_category = "HateSpeech"
                    st.session_state.is_formalized_comment = False  # Reset para la próxima vez
            
            # 4. Extraer tags del comentario final
            extracted_tags = extract_tags_from_text(final_comment, available_tags)
            
            # 5. Crear estructura de datos
            analysis_data = {
                "timestamp": datetime.now().isoformat(),
                "comentario": final_comment,
                "categoria": final_category,
                "tags": extracted_tags
            }
            
            # 6. Guardar en JSON
            if save_to_json(analysis_data):
                st.success("✅ Análisis guardado exitosamente")
            
            # 7. Mostrar resultados
            st.divider()
            st.subheader("📋 Resultados del Análisis:")
            
            # Mostrar categoría con color
            category_colors = {
                "Sugerencia": "🟢",
                "Opinion": "🔵", 
                "HateSpeech": "🔴",
                "Vida universitaria": "🟡"
            }
            
            st.markdown(f"**Categoría:** {category_colors.get(final_category, '⚪')} **{final_category}**")
            
            # Mostrar tags
            if extracted_tags:
                st.markdown(f"**Tags encontrados:** {', '.join(extracted_tags)}")
            else:
                st.markdown("**Tags encontrados:** Ninguno")
            
            # Mostrar JSON generado
            with st.expander("Ver JSON generado"):
                st.json(analysis_data)
    
    elif analyze_button:
        st.warning("⚠️ Se tiene que escribir algo coherente")
    
    # Mostrar historial reciente
    if history:
        st.divider()
        st.subheader("📚 Historial Reciente (últimos 5 análisis)")
        
        for item in reversed(history[-5:]):  # Mostrar los últimos 5
            # Manejar tanto estructura antigua como nueva
            comentario_display = item.get('comentario', item.get('comentario_final', 'Sin comentario'))
            comentario_preview = comentario_display[:50] + "..." if len(comentario_display) > 50 else comentario_display
            
            with st.expander(f"{item['categoria']} - {comentario_preview}"):
                st.write(f"**Comentario:** {comentario_display}")
                st.write(f"**Categoría:** {item['categoria']}")
                st.write(f"**Tags:** {', '.join(item['tags']) if item['tags'] else 'Ninguno'}")
                st.write(f"**Fecha:** {item['timestamp']}")
                
                st.write(f"**Categoría:** {item['categoria']}")
                st.write(f"**Tags:** {', '.join(item['tags']) if item['tags'] else 'Ninguno'}")
                st.write(f"**Fecha:** {item['timestamp']}")

if __name__ == "__main__":
    main()
