import os
import sys
import json
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configuración para Windows y PyTorch
if sys.platform == "win32":
    try:
        import torch
        torch.classes.__path__ = []
    except ImportError:
        pass  

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Cargar variables de entorno
load_dotenv()

# Configurar Flask
app = Flask(__name__)
CORS(app) 

# Configurar claves API
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("Error: No se encontró la clave API de Groq. Asegúrate de configurar GROQ_API_KEY en tu archivo .env")
    sys.exit(1)

model = ChatGroq(groq_api_key=groq_api_key, model="llama-3.3-70b-versatile")

def load_tags():
    """Cargar tags desde el archivo tags.txt"""
    try:
        with open("tags.txt", "r", encoding="utf-8") as file:
            tags = [line.strip().lower() for line in file if line.strip()]
        return tags
    except FileNotFoundError:
        print("Warning: No se encontró el archivo tags.txt")
        return []
    except Exception as e:
        print(f"Error cargando tags: {str(e)}")
        return []

def is_coherent_text(text):
    """Validar si el texto es coherente (más de una palabra y tiene sentido básico) - Soporta español e inglés"""
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
    
    # 5. Detectar incoherencia semántica usando reglas híbridas (español e inglés)
    
    # Lista de patrones específicamente incoherentes (español)
    incoherent_patterns_es = [
        "casa azul mojado", "perro volando matemáticas", "mesa correr feliz",
        "computadora cantar verde", "silla bailar número", "árbol escribir calor",
        "teléfono dormir azúcar", "libro nadar rojo", "ventana comer fríos",
        "zapato volar música", "reloj bailar agua", "puerta correr números"
    ]
    
    # Lista de patrones incoherentes (inglés)
    incoherent_patterns_en = [
        "house blue wet", "dog flying mathematics", "table run happy",
        "computer sing green", "chair dance number", "tree write heat",
        "phone sleep sugar", "book swim red", "window eat cold",
        "shoe fly music", "clock dance water", "door run numbers"
    ]
    
    text_lower = text.lower().strip()
    if text_lower in incoherent_patterns_es or text_lower in incoherent_patterns_en:
        return False
    
    # Verificar patrones de incoherencia (sustantivo + verbo incongruente + adjetivo/sustantivo)
    if len(words) == 3:
        physical_objects = {
            # Español
            'casa', 'mesa', 'computadora', 'silla', 'árbol', 'teléfono', 'libro', 'ventana', 'zapato', 'reloj', 'puerta',
            # Inglés
            'house', 'table', 'computer', 'chair', 'tree', 'phone', 'book', 'window', 'shoe', 'clock', 'door'
        }
        action_verbs = {
            # Español
            'cantar', 'bailar', 'correr', 'nadar', 'volar', 'escribir', 'dormir', 'comer',
            # Inglés
            'sing', 'dance', 'run', 'swim', 'fly', 'write', 'sleep', 'eat'
        }
        if words[0].lower() in physical_objects and words[1].lower() in action_verbs:
            return False
    
    # Verificar estructura mínima de oración (español E inglés)
    structure_indicators = {
        # ESPAÑOL
        # Artículos
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
        # Verbos auxiliares/comunes
        'es', 'está', 'son', 'están', 'tiene', 'tienen', 'hay', 'fue', 'era',
        # Preposiciones
        'de', 'del', 'en', 'con', 'por', 'para', 'desde', 'hasta', 'sobre',
        # Pronombres
        'que', 'se', 'me', 'te', 'le', 'nos', 'les', 'mi', 'tu', 'su',
        # Adverbios comunes
        'muy', 'más', 'menos', 'bien', 'mal', 'no', 'sí', 'y', 'o', 'pero',
        # Verbos modales/comunes
        'quiero', 'necesito', 'creo', 'pienso', 'siento', 'veo', 'escucho',
        'necesitamos', 'queremos', 'podemos', 'debemos',
        
        # INGLÉS
        # Articles
        'the', 'a', 'an',
        # Common verbs
        'is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did',
        # Prepositions
        'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'about', 'of',
        # Pronouns
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'our', 'their',
        # Common adverbs
        'very', 'more', 'less', 'well', 'badly', 'not', 'yes', 'and', 'or', 'but',
        # Modal/common verbs
        'want', 'need', 'think', 'feel', 'see', 'hear', 'can', 'could', 'should', 'would'
    }
    
    text_words = set(word.lower() for word in words)
    has_structure = bool(text_words.intersection(structure_indicators))
    
    if not has_structure:
        return False
    
    return True

def extract_tags_from_text(text, available_tags):
    """Extraer tags relevantes del texto - cantidad dinámica según longitud"""
    text_lower = text.lower()
    tag_scores = {}  
    # Palabras del texto
    words_in_text = re.findall(r'\b\w+\b', text_lower)
    
    # Determinar cantidad máxima de tags según longitud del texto
    if len(words_in_text) <= 4:  
        max_tags = 1
    elif len(words_in_text) <= 8: 
        max_tags = 2
    else:  
        max_tags = 3
    
    for tag in available_tags:
        score = 0
        matches = []  
        
        # 1. Búsqueda exacta
        for i, word in enumerate(words_in_text):
            if word == tag:
                matches.append(i)
                score += 3  
        
        # 2. Búsqueda de variaciones simples
        if not matches: 
            for i, word in enumerate(words_in_text):
                if word == tag + 's' or tag == word + 's':
                    matches.append(i)
                    score += 2  
                    break
                elif word == tag + 'es' or tag == word + 'es':
                    matches.append(i)
                    score += 2
                    break
        
        # Si encontramos el tag, calcular puntuación final
        if matches:
            score += len(tag) * 0.1
            first_position = min(matches)
            position_bonus = max(0, 5 - first_position)
            score += position_bonus
            frequency_bonus = len(matches) * 0.5
            score += frequency_bonus
            tag_scores[tag] = score
    
    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    most_relevant_tags = [tag for tag, score in sorted_tags[:max_tags]]
    
    return most_relevant_tags

def categorize_comment(comment):
    """Categorize the comment using LLM - English only"""
    template = """
    Analyze the following comment and categorize it EXACTLY into one of these four categories:
    - "Suggestion": If the comment proposes improvements, ideas, changes or constructive recommendations
    - "Opinion": If the comment expresses a neutral or positive personal opinion, experiences without being offensive
    - "Complaint": If the comment contains offensive language, discrimination, threats, insults, very negative criticism, or very negative feelings towards people or places (e.g.: "this place is bullshit", "the teacher sucks", "I hate...", "it's terrible", etc.)
    - "University life": If the comment specifically refers to experiences, situations, activities or aspects of university, academic or student life that do not fit into the other categories

    Important rules:
    1. Respond ONLY with one of these four words: "Suggestion", "Opinion", "Complaint", or "University life"
    2. Do not add additional explanations
    3. Negative comments about people (teachers, classmates, etc.) or places go in "Complaint"
    4. Comments about classes, university, studies, campus, etc. go in "University life"
    5. If in doubt, prioritize in this order: Complaint > University life > Suggestion > Opinion
    6. Words like "bullshit", "sucks", "terrible", "awful" are ALWAYS "Complaint"

    Examples:
    - "The teacher is terrible" → Complaint
    - "This place is bullshit" → Complaint
    - "Math class is challenging" → University life
    - "They should improve the cafeteria" → Suggestion
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
        valid_categories = ["Suggestion", "Opinion", "Complaint", "University life"]
        
        if category in valid_categories:
            return category
        else:
            for valid_cat in valid_categories:
                if valid_cat.lower() in category.lower():
                    return valid_cat
            return "Opinion"
            
    except Exception as e:
        print(f"Error categorizando comentario: {str(e)}")
        return "Opinion"

def formalize_hate_speech(comment):
    """Convert offensive comment to formal and appropriate language - English only"""
    template = """
    You are an English language moderator. The following comment contains offensive language. Convert it to a formal, respectful and constructive comment that expresses the same idea but in a manner appropriate for an academic or professional environment.

    CRITICAL RULES:
    1. Remove all offensive words, vulgarities or insults (e.g., "bullshit", "sucks", "terrible", "awful", etc.)
    2. Maintain the essence of the message but in a constructive tone
    3. Use formal and respectful English language
    4. If it's a complaint, convert it to constructive feedback
    5. Maximum 300 characters
    6. Respond ONLY with the formalized text in ENGLISH, without additional explanations
    7. **MANDATORY**: Always respond in ENGLISH only. Never use Spanish or any other language.

    Examples:
    - Original: "This teacher sucks" → Formalized: "I believe the teaching methodology could be improved"
    - Original: "This place is bullshit" → Formalized: "I think this location has areas that could benefit from improvement"
    - Original: "The cafeteria is awful" → Formalized: "The cafeteria services could be enhanced"

    Original comment: "{comment}"

    Formalized comment (ENGLISH ONLY):
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    try:
        response = ""
        for chunk in chain.stream({"comment": comment}):
            response += chunk.content
        
        formalized = response.strip()
        
        if formalized.startswith('"') and formalized.endswith('"'):
            formalized = formalized[1:-1]
        elif formalized.startswith("'") and formalized.endswith("'"):
            formalized = formalized[1:-1]
        
        formalized = formalized.replace('""', '').replace("''", '')
        
        if len(formalized) < 10:
            formalized = "Comentario convertido a lenguaje apropiado por contener contenido ofensivo."
        
        return formalized
        
    except Exception as e:
        print(f"Error formalizando comentario: {str(e)}")
        return "Comentario modificado por contener contenido inapropiado."

def save_to_json(data, filename="comentarios_analizados.json"):
    """Guardar datos en archivo JSON"""
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
        else:
            existing_data = []
        
        existing_data.append(data)
        
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Error guardando en JSON: {str(e)}")
        return False

def load_analysis_history():
    """Cargar historial de análisis desde JSON"""
    try:
        if os.path.exists("comentarios_analizados.json"):
            with open("comentarios_analizados.json", "r", encoding="utf-8") as file:
                return json.load(file)
        return []
    except Exception as e:
        print(f"Error cargando historial: {str(e)}")
        return []

# Cargar tags disponibles
available_tags = load_tags()

def analyze_title(title):
    """Analizar título: verificar coherencia, detectar contenido ofensivo y dar recomendación"""
    
    # 1. Verificar coherencia básica del título
    is_coherent = is_coherent_text(title)
    
    # 2. Detectar si es ofensivo usando IA
    offensive_template = """
    Analiza el siguiente título y determina si contiene contenido ofensivo, discriminatorio, vulgar o inapropiado.
    
    Responde EXACTAMENTE con una de estas dos palabras:
    - "OFENSIVO": Si contiene insultos, discriminación, vulgaridades, lenguaje de odio o contenido inapropiado
    - "APROPIADO": Si es un título normal y apropiado
    
    Título: "{title}"
    
    Clasificación:
    """
    
    prompt = ChatPromptTemplate.from_template(offensive_template)
    chain = prompt | model
    
    try:
        response = ""
        for chunk in chain.stream({"title": title}):
            response += chunk.content
        
        is_offensive = "OFENSIVO" in response.strip().upper()
        
    except Exception as e:
        print(f"Error detectando contenido ofensivo: {str(e)}")
        is_offensive = False
    
    # 3. Generar recomendación automática basada en el análisis
    recommendation = None
    titulo_sugerido = None
    
    if not is_coherent:
        # Si no es coherente, sugerir que lo reformule
        recommendation = "El título necesita ser más claro y comprensible."
        titulo_sugerido = None
        
    elif is_offensive:
        # Si es ofensivo, generar automáticamente una versión apropiada
        fix_template = """
        El siguiente título contiene contenido ofensivo. Genera una versión alternativa que sea:
        1. Respetuosa y apropiada
        2. Mantenga la esencia del mensaje original
        3. Sea clara y profesional
        4. Máximo 50 caracteres
        
        Título ofensivo: "{title}"
        
        Responde SOLO con el título corregido, sin explicaciones adicionales.
        
        Título corregido:
        """
        
        prompt = ChatPromptTemplate.from_template(fix_template)
        chain = prompt | model
        
        try:
            response = ""
            for chunk in chain.stream({"title": title}):
                response += chunk.content
            
            titulo_sugerido = response.strip()
            
            # Limpiar comillas si las tiene
            if titulo_sugerido.startswith('"') and titulo_sugerido.endswith('"'):
                titulo_sugerido = titulo_sugerido[1:-1]
            elif titulo_sugerido.startswith("'") and titulo_sugerido.endswith("'"):
                titulo_sugerido = titulo_sugerido[1:-1]
            
            recommendation = f"Título corregido automáticamente"
            
        except Exception as e:
            print(f"Error generando título corregido: {str(e)}")
            titulo_sugerido = "Título modificado por contener contenido inapropiado"
            recommendation = "Título corregido automáticamente"
    
    else:
        # Si es coherente y apropiado, dar validación positiva
        recommendation = "Título apropiado y coherente"
        titulo_sugerido = None
    
    return {
        "is_coherent": is_coherent,
        "is_offensive": is_offensive,
        "recommendation": recommendation,
        "titulo_sugerido": titulo_sugerido,  # Nueva propiedad
        "status": "apropiado" if (is_coherent and not is_offensive) else "requiere_revision"
    }


# RUTAS DE LA API

@app.route("/")
def home():
    return jsonify({"message": "Servidor activo"}), 200


# Variable global para almacenar el comentario actual
comentario_actual = None

@app.route('/comentario', methods=['POST'])
def obtener_comentario():
    """Endpoint para recibir un comentario y establecerlo como comentario actual"""
    global comentario_actual
    
    try:
        data = request.get_json() if request.is_json else {}
        
        # Extraer el comentario de data
        if 'comentario' in data:
            comentario_actual = data['comentario']
        elif 'text' in data:
            comentario_actual = data['text']
        elif isinstance(data, str):
            comentario_actual = data
        else:
            # Si data tiene solo un valor string, usarlo
            if len(data) == 1:
                comentario_actual = list(data.values())[0]
            else:
                comentario_actual = str(data)
        
        return jsonify({
            "success": True,
            "data": {
                "comentario": comentario_actual,
                "data_recibida": data  # Para debug
            },
            "message": "Comentario recibido y guardado exitosamente"
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Error interno del servidor: {str(e)}"
        }), 500

@app.route('/comentario/actual', methods=['GET'])
def obtener_comentario_actual():
    """Endpoint para obtener el comentario actual guardado"""
    global comentario_actual
    
    if comentario_actual is None:
        return jsonify({
            "error": "No hay comentario disponible en este momento"
        }), 404
    
    return jsonify({
        "success": True,
        "data": {
            "comentario": comentario_actual
        }
    })

@app.route('/comentario', methods=['GET'])
def procesar_comentario_actual():
    """Endpoint para procesar automáticamente el comentario actual con IA"""
    global comentario_actual
    
    try:
        # Verificar que hay un comentario actual
        if comentario_actual is None:
            return jsonify({
                "error": "No hay comentario disponible para procesar. Envía primero un comentario con POST."
            }), 404
        
        # Procesar el comentario con IA completo
        
        # 1. Validar coherencia del texto
        is_coherent = is_coherent_text(comentario_actual)
        
        if not is_coherent:
            return jsonify({
                "error": "El comentario no es coherente o no tiene suficiente contenido válido",
                "comentario_recibido": comentario_actual
            }), 400
        
        # 2. Categorizar el comentario
        categoria = categorize_comment(comentario_actual)
        
        # 3. Extraer tags
        tags = extract_tags_from_text(comentario_actual, available_tags)
        
        # 4. Si es una queja (hate speech), formalizarlo
        comentario_final = comentario_actual
        comentario_formalizado = None
        
        if categoria == "Complaint":
            comentario_formalizado = formalize_hate_speech(comentario_actual)
        
        # Crear el análisis completo
        analysis_data = {
            "id": len(load_analysis_history()) + 1,
            "timestamp": datetime.now().isoformat(),
            "comentario_original": comentario_actual,
            "comentario_formalizado": comentario_formalizado,  # Solo si es Queja
            "categoria": categoria,
            "tags": tags,
            "is_coherent": is_coherent,
            "is_offensive": categoria == "Complaint"  # TRUE si es ofensivo
        }
        
        # Guardar el análisis
        if save_to_json(analysis_data):
            # Limpiar el comentario actual después de procesarlo
            comentario_actual = None
            
            return jsonify({
                "success": True,
                "data": analysis_data,
                "message": "Comentario procesado y analizado exitosamente"
            })
        else:
            return jsonify({
                "error": "Error al guardar el análisis"
            }), 500
            
    except Exception as e:
        return jsonify({
            "error": f"Error interno del servidor: {str(e)}"
        }), 500



@app.route('/procesar', methods=['POST'])
def recibir_y_procesar_comentario():
    """Endpoint que recibe un comentario y lo procesa automáticamente con IA en una sola llamada"""
    global comentario_actual
    
    try:
        # 1. RECIBIR EL COMENTARIO (igual que el POST /comentario)
        data = request.get_json() if request.is_json else {}
        
        # Extraer el comentario de data
        if 'comentario' in data:
            comentario_recibido = data['comentario']
        elif 'text' in data:
            comentario_recibido = data['text']
        elif isinstance(data, str):
            comentario_recibido = data
        else:
            # Si data tiene solo un valor string, usarlo
            if len(data) == 1:
                comentario_recibido = list(data.values())[0]
            else:
                comentario_recibido = str(data)
        
        if not comentario_recibido or comentario_recibido.strip() == "":
            return jsonify({
                "error": "Se requiere un comentario válido"
            }), 400
        
        # Establecer como comentario actual temporalmente
        comentario_actual = comentario_recibido.strip()
        
        # 2. PROCESAR EL COMENTARIO (igual que el GET /comentario)
        
        # 2.1. Validar coherencia del texto
        is_coherent = is_coherent_text(comentario_actual)
        
        if not is_coherent:
            # Limpiar comentario actual si no es coherente
            comentario_actual = None
            return jsonify({
                "error": "El comentario no es coherente o no tiene suficiente contenido válido",
                "comentario_recibido": comentario_recibido
            }), 400
        
        # 2.2. Categorizar el comentario
        categoria = categorize_comment(comentario_actual)
        
        # 2.3. Extraer tags
        tags = extract_tags_from_text(comentario_actual, available_tags)
        
        # 2.4. Si es una queja (hate speech), formalizarlo
        comentario_formalizado = None
        
        if categoria == "Complaint":
            comentario_formalizado = formalize_hate_speech(comentario_actual)
        
        # 2.5. Crear el análisis completo
        analysis_data = {
            "id": len(load_analysis_history()) + 1,
            "timestamp": datetime.now().isoformat(),
            "comentario_original": comentario_actual,
            "comentario_formalizado": comentario_formalizado,  # Solo si es Queja
            "categoria": categoria,
            "tags": tags,
            "is_coherent": is_coherent,
            "is_offensive": categoria == "Complaint"  # TRUE si es ofensivo
        }
        
        # 2.6. Guardar el análisis
        if save_to_json(analysis_data):
            # Limpiar el comentario actual después de procesarlo
            comentario_actual = None
            
            return jsonify({
                "success": True,
                "data": analysis_data,
                "message": "Comentario recibido, procesado y analizado exitosamente en una sola operación"
            })
        else:
            comentario_actual = None
            return jsonify({
                "error": "Error al guardar el análisis"
            }), 500
            
    except Exception as e:
        comentario_actual = None
        return jsonify({
            "error": f"Error interno del servidor: {str(e)}"
        }), 500

@app.route('/procesartitulos', methods=['POST'])
def procesar_titulo():
    """Endpoint para analizar títulos: verificar coherencia, detectar contenido ofensivo y dar recomendación"""
    
    try:
        data = request.get_json() if request.is_json else {}
        
        # Extraer el título de data
        if 'titulo' in data:
            titulo = data['titulo']
        elif 'title' in data:
            titulo = data['title']
        elif 'comentario' in data:  # Por compatibilidad
            titulo = data['comentario']
        elif isinstance(data, str):
            titulo = data
        else:
            # Si data tiene solo un valor string, usarlo
            if len(data) == 1:
                titulo = list(data.values())[0]
            else:
                return jsonify({
                    "error": "Se requiere un campo 'titulo' o 'title' en el JSON"
                }), 400
        
        if not titulo or titulo.strip() == "":
            return jsonify({
                "error": "El título no puede estar vacío"
            }), 400
        
        titulo = titulo.strip()
        
        # Analizar el título usando la función específica
        analysis_result = analyze_title(titulo)
        
        # Crear respuesta completa
        response_data = {
            "id": len(load_analysis_history()) + 1,
            "timestamp": datetime.now().isoformat(),
            "titulo_original": titulo,
            "es_coherente": analysis_result["is_coherent"],
            "es_ofensivo": analysis_result["is_offensive"],
            "recomendacion": analysis_result["recommendation"],
            "titulo_sugerido": analysis_result["titulo_sugerido"],  # Nuevo campo
            "estado": analysis_result["status"]
        }
        
        # Guardar el análisis en el historial
        save_to_json(response_data, "titulos_analizados.json")
        
        return jsonify({
            "success": True,
            "data": response_data,
            "message": "Título analizado exitosamente"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Error interno del servidor: {str(e)}"
        }), 500

# Manejo de errores globales
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint no encontrado"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Método no permitido para este endpoint"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Error interno del servidor"
    }), 500

if __name__ == '__main__':
    # Ejecutar la aplicación
    app.run(debug=True, host='0.0.0.0', port=5000)