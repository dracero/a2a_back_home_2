import os
import json
import time
import asyncio
import torch
import glob
from PyPDF2 import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import AutoTokenizer, AutoModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from autogen import AssistantAgent, UserProxyAgent
from typing import Dict, Any, AsyncIterable, List
import logging
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
logger = logging.getLogger(__name__)

class PhysicsAgent:
    """Agente de física con capacidades de streaming y invoke para integración con task_manager"""
    
    SUPPORTED_CONTENT_TYPES = ["text/plain"]
    
    def __init__(self, auto_load_pdfs: bool = True):
        # Configurar APIs
        self._setup_apis()
        
        # Inicializar componentes principales
        self.llm = None  # LLM principal unificado
        self.memoria_semantica = None
        self.agents = {}
        self.temario = ""
        self.contenido_completo = ""
        self.sessions = {}  # Para manejar sesiones de conversación
        
        # Configuración de embedding
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = None
        self.model = None
        
        # Configuración de Qdrant
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_KEY")
        self.collection_name = "documentos_pdf"
        
        # Configuración de PDFs
        self.pdf_dir = os.getenv("PDF_DIR", "/home/cetec/Downloads/apuntes_fisica")
        self.pdf_files = []
        self.pdfs_processed = False
        
        # Configuración de Hugging Face
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        
        # Inicializar componentes al crear la instancia
        self.inicializar_componentes()
        
        # Cargar PDFs automáticamente si se especifica
        if auto_load_pdfs:
            self.cargar_pdfs_automaticamente()
        
        print("✅ PhysicsAgent inicializado correctamente")

    def _setup_apis(self):
        """Configurar las APIs necesarias"""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        os.environ["GOOGLE_API_KEY"] = google_api_key
        print("✅ APIs configuradas")

    def inicializar_componentes(self):
        """Inicializar todos los componentes del asistente"""
        self._inicializar_llm()
        self._inicializar_memoria()
        self._inicializar_agentes()
        self._inicializar_modelo_embedding()
        print("✅ Todos los componentes inicializados")

    def _inicializar_llm(self):
        """Inicializar el modelo de lenguaje unificado"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
            top_p=0.95,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            convert_system_message_to_human=True
        )
        print("✅ LLM unificado inicializado")

    def _inicializar_memoria(self):
        """Inicializar la memoria semántica"""
        self.memoria_semantica = self.SemanticMemory(llm=self.llm)
        print("✅ Memoria semántica inicializada")

    def _inicializar_agentes(self):
        """Inicializar los agentes de AutoGen"""
        config_list = [{
            "model": "gemini-2.5-flash",
            "api_type": "google",
            "api_key": os.getenv("GOOGLE_API_KEY")
        }]

        llm_config = {
            "config_list": config_list,
            "timeout": 120,
            "temperature": 0,
            "seed": 42
        }

        self.agents['user_proxy'] = UserProxyAgent(
            name="UserProxy",
            system_message="Eres un asistente que recibe las consultas de los usuarios sobre física y las transmite al clasificador para su procesamiento.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )

        self.agents['classifier'] = AssistantAgent(
            name="Clasificador",
            system_message=f"""Eres un agente especializado en clasificar consultas de física según el temario proporcionado.
            Debes proporcionar:
            1. El número y título del tema principal
            2. Los subtemas relevantes
            3. Palabras clave para búsqueda
            Formato:
            TEMA: [número y título]
            SUBTEMAS: [lista]
            KEYWORDS: [palabras clave]

            TEMARIO DE FÍSICA:
            {self.temario}
            """,
            llm_config=llm_config
        )

        self.agents['search'] = AssistantAgent(
            name="BuscadorQdrant",
            system_message="""Eres un agente de búsqueda especializado en física. Recibes clasificaciones y las conviertes en consultas efectivas para buscar en documentos.
            Proporciona una consulta clara y específica que pueda usarse para buscar en la base de conocimientos.""",
            llm_config=llm_config
        )

        self.agents['response'] = AssistantAgent(
            name="RespondeConsulta",
            system_message="""Eres un profesor de física que responde consultas. Utiliza la información del clasificador y los resultados de búsqueda.
            Proporciona respuestas claras, estructuradas y basadas en evidencia.""",
            llm_config=llm_config
        )

        print("✅ Agentes inicializados")

    def _inicializar_modelo_embedding(self):
        """Inicializar el modelo de embeddings"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=os.getenv("HF_TOKEN") if os.getenv("HF_TOKEN") else None
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                token=os.getenv("HF_TOKEN") if os.getenv("HF_TOKEN") else None
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            print("✅ Modelo de embeddings inicializado")
        except Exception as e:
            logger.error(f"Error al inicializar modelo de embeddings: {e}")
            print("⚠️ Modelo de embeddings no disponible, funcionalidad limitada")

    def _get_session_memory(self, session_id: str):
        """Obtener o crear memoria para una sesión específica"""
        if session_id not in self.sessions:
            self.sessions[session_id] = self.SemanticMemory(llm=self.llm)
        return self.sessions[session_id]

    # Métodos principales para integración A2A
    def invoke(self, query: str, session_id: str = "default") -> str:
        """Método invoke para procesamiento síncrono"""
        try:
            if not self.pdfs_processed:
                return "⚠️ Los PDFs no han sido procesados aún. Por favor, espera a que se complete la inicialización."
            
            # Ejecutar el flujo asíncrono de forma síncrona
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._process_query(query, session_id))
                return result
            finally:
                loop.close()
                
        except Exception as e:
            error_msg = f"Error en invoke: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def stream(self, query: str, session_id: str = "default") -> AsyncIterable[Dict[str, Any]]:
        """Método stream para procesamiento asíncrono con actualizaciones en tiempo real"""
        try:
            if not self.pdfs_processed:
                yield {
                    "is_task_complete": True,
                    "content": "⚠️ Los PDFs no han sido procesados aún. Por favor, espera a que se complete la inicialización o agrega PDFs manualmente."
                }
                return
            
            # Enviar actualización inicial
            yield {
                "is_task_complete": False,
                "updates": "🔄 Iniciando procesamiento de la consulta..."
            }
            
            # Obtener memoria de la sesión
            session_memory = self._get_session_memory(session_id)
            contexto_memoria = session_memory.get_context()
            
            # Clasificación
            yield {
                "is_task_complete": False,
                "updates": "🔍 Clasificando la consulta..."
            }
            
            clasificacion_respuesta = self.agents['classifier'].generate_reply(messages=[{
                "role": "user",
                "content": f"Consulta: {query}\n\nContexto de la conversación: {contexto_memoria}"
            }])
            clasificacion = clasificacion_respuesta.get("content", "")
            
            yield {
                "is_task_complete": False,
                "updates": f"✅ Clasificación completada: {clasificacion[:100]}..."
            }
            
            # Generación de consulta para búsqueda
            yield {
                "is_task_complete": False,
                "updates": "🔎 Generando consulta de búsqueda..."
            }
            
            consulta_busqueda_respuesta = self.agents['search'].generate_reply(messages=[{
                "role": "user",
                "content": f"Clasificación: {clasificacion}\n\nConsulta original: {query}\n\nContexto de la conversación: {contexto_memoria}"
            }])
            consulta_busqueda = consulta_busqueda_respuesta.get("content", "")
            
            # Realizar búsqueda en Qdrant
            yield {
                "is_task_complete": False,
                "updates": "📚 Buscando en la base de conocimientos..."
            }
            
            resultados_busqueda = await self.search_documents(consulta_busqueda)
            
            yield {
                "is_task_complete": False,
                "updates": f"✅ Encontrados {len(resultados_busqueda)} documentos relevantes"
            }
            
            # Generación de respuesta final
            yield {
                "is_task_complete": False,
                "updates": "🤖 Generando respuesta final..."
            }
            
            respuesta_final = await self.generar_respuesta_final(
                query, resultados_busqueda, clasificacion, contexto_memoria
            )
            
            # Actualizar memoria de la sesión
            session_memory.add_interaction(query, respuesta_final)
            
            # Enviar respuesta final
            yield {
                "is_task_complete": True,
                "content": respuesta_final
            }
            
        except Exception as e:
            error_msg = f"Error en stream: {str(e)}"
            logger.error(error_msg)
            yield {
                "is_task_complete": True,
                "content": error_msg
            }

    async def _process_query(self, query: str, session_id: str = "default") -> str:
        """Procesar consulta de forma asíncrona (usado por invoke)"""
        try:
            # Obtener memoria de la sesión
            session_memory = self._get_session_memory(session_id)
            contexto_memoria = session_memory.get_context()
            
            # Clasificación
            clasificacion_respuesta = self.agents['classifier'].generate_reply(messages=[{
                "role": "user",
                "content": f"Consulta: {query}\n\nContexto de la conversación: {contexto_memoria}"
            }])
            clasificacion = clasificacion_respuesta.get("content", "")
            
            # Generación de consulta para búsqueda
            consulta_busqueda_respuesta = self.agents['search'].generate_reply(messages=[{
                "role": "user",
                "content": f"Clasificación: {clasificacion}\n\nConsulta original: {query}\n\nContexto de la conversación: {contexto_memoria}"
            }])
            consulta_busqueda = consulta_busqueda_respuesta.get("content", "")
            
            # Realizar búsqueda en Qdrant
            resultados_busqueda = await self.search_documents(consulta_busqueda)
            
            # Generación de respuesta final
            respuesta_final = await self.generar_respuesta_final(
                query, resultados_busqueda, clasificacion, contexto_memoria
            )
            
            # Actualizar memoria de la sesión
            session_memory.add_interaction(query, respuesta_final)
            
            return respuesta_final
            
        except Exception as e:
            error_msg = f"Error al procesar consulta: {str(e)}"
            logger.error(error_msg)
            return error_msg

    # Funciones para procesamiento de PDFs
    def cargar_pdfs_automaticamente(self):
        """Cargar PDFs automáticamente del directorio especificado"""
        try:
            if not os.path.exists(self.pdf_dir):
                print(f"⚠️ Directorio de PDFs no encontrado: {self.pdf_dir}")
                return
            
            # Buscar archivos PDF en el directorio
            self.pdf_files = glob.glob(os.path.join(self.pdf_dir, "*.pdf"))
            
            if not self.pdf_files:
                print(f"⚠️ No se encontraron archivos PDF en: {self.pdf_dir}")
                return
            
            print(f"📚 Encontrados {len(self.pdf_files)} archivos PDF:")
            for pdf in self.pdf_files:
                print(f"  - {os.path.basename(pdf)}")
            
            # Procesar PDFs para extraer temario
            print("🔄 Procesando PDFs para extraer temario...")
            self.procesar_pdfs_temario(self.pdf_files)
            
            # Procesar y almacenar PDFs en Qdrant (asíncrono)
            print("🔄 Procesando PDFs para almacenar en Qdrant...")
            asyncio.run(self.procesar_y_almacenar_pdfs(self.pdf_files))
            
            self.pdfs_processed = True
            print("✅ PDFs procesados y almacenados correctamente")
            
        except Exception as e:
            print(f"❌ Error al cargar PDFs automáticamente: {e}")
            logger.error(f"Error al cargar PDFs: {e}")

    def agregar_pdfs_manuales(self, pdf_paths: List[str]):
        """Agregar PDFs manualmente"""
        try:
            pdf_paths_validos = []
            for pdf_path in pdf_paths:
                if os.path.exists(pdf_path) and pdf_path.endswith('.pdf'):
                    pdf_paths_validos.append(pdf_path)
                    print(f"✅ PDF válido: {os.path.basename(pdf_path)}")
                else:
                    print(f"⚠️ PDF no válido o no encontrado: {pdf_path}")
            
            if not pdf_paths_validos:
                print("❌ No se encontraron PDFs válidos")
                return False
            
            # Agregar a la lista existente
            self.pdf_files.extend(pdf_paths_validos)
            self.pdf_files = list(set(self.pdf_files))  # Eliminar duplicados
            
            # Procesar los nuevos PDFs
            print("🔄 Procesando nuevos PDFs...")
            self.procesar_pdfs_temario(pdf_paths_validos)
            asyncio.run(self.procesar_y_almacenar_pdfs(pdf_paths_validos))
            
            self.pdfs_processed = True
            print("✅ Nuevos PDFs procesados correctamente")
            return True
            
        except Exception as e:
            print(f"❌ Error al agregar PDFs manuales: {e}")
            logger.error(f"Error al agregar PDFs manuales: {e}")
            return False

    def leer_pdf(self, nombre_archivo):
        """Leer contenido de un archivo PDF"""
        try:
            reader = PdfReader(nombre_archivo)
            texto = ""
            for page in reader.pages:
                texto += page.extract_text()
            return texto
        except Exception as e:
            print(f"Error al leer {nombre_archivo}: {e}")
            return ""

    def procesar_pdfs_temario(self, archivos_pdf):
        """Procesar PDFs para extraer el temario"""
        contenido_completo = ""

        for archivo in archivos_pdf:
            try:
                contenido_completo += f"\n--- Contenido de {os.path.basename(archivo)} ---\n"
                contenido_completo += self.leer_pdf(archivo)
            except Exception as e:
                print(f"Error al procesar {archivo}: {e}")

        self.contenido_completo = contenido_completo

        # Extraer temario usando el LLM unificado
        system_message = f"""
        Eres un experto profesor Física I de la Universidad de Buenos Aires.
        Tu tarea es responder preguntas sobre el temario que tiene en los archivos que lees, proporcionando explicaciones claras, detalladas y ejemplos relevantes.
        Responde solo con el contenido, si no está en el contenido di que no tienes eso en tu base de datos.
        Utiliza el siguiente contenido como referencia para tus respuestas:
        ---
        {contenido_completo}
        ---
        """

        user_question = "Sobre que contenidos podes contestarme"

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_question),
        ]

        try:
            ai_msg = self.llm.invoke(messages)
            self.temario = ai_msg.content
            print("✅ Temario extraído correctamente")
            print("\nTEMARIO EXTRAÍDO:\n" + self.temario + "\n")
        except Exception as e:
            print(f"Error al extraer temario: {e}")
            self.temario = "Error al extraer temario"

        return self.temario

    def split_into_chunks(self, text, chunk_size=2000):
        """Dividir texto en chunks"""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def generate_embeddings(self, chunks, batch_size=32):
        """Generar embeddings para los chunks"""
        if not self.model or not self.tokenizer:
            logger.warning("Modelo de embeddings no disponible")
            return []
            
        embeddings = []
        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings.extend(outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy())
        except Exception as e:
            print(f"Error al generar embeddings: {e}")
        return embeddings

    async def store_in_qdrant(self, points):
        """Almacenar puntos en Qdrant"""
        try:
            client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

            # Crear colección si no existe
            try:
                await client.get_collection(self.collection_name)
            except:
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=len(points[0].vector), distance=Distance.COSINE)
                )
                print(f"Colección '{self.collection_name}' creada")

            # Insertar datos
            await client.upsert(collection_name=self.collection_name, points=points)
            print(f"{len(points)} chunks almacenados en Qdrant")
        except Exception as e:
            print(f"Error al almacenar en Qdrant: {e}")

    async def procesar_y_almacenar_pdfs(self, pdf_files):
        """Procesar PDFs y almacenar en Qdrant"""
        all_chunks = []
        pdf_metadata = []

        for pdf_file in pdf_files:
            if not os.path.exists(pdf_file):
                print(f"⚠️ {pdf_file} no encontrado")
                continue

            try:
                print(f"📖 Procesando: {os.path.basename(pdf_file)}")
                
                # Procesar PDF
                text = self.leer_pdf(pdf_file)
                if not text:
                    print(f"⚠️ No se pudo extraer texto de: {os.path.basename(pdf_file)}")
                    continue
                    
                chunks = self.split_into_chunks(text)
                embeddings = self.generate_embeddings(chunks)

                if not embeddings:
                    print(f"⚠️ No se pudieron generar embeddings para: {os.path.basename(pdf_file)}")
                    continue

                # Registrar metadatos
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    all_chunks.append(chunk)
                    pdf_metadata.append({
                        "pdf_name": os.path.basename(pdf_file),
                        "pdf_path": pdf_file,
                        "chunk_id": i
                    })
                    
                print(f"✅ {os.path.basename(pdf_file)}: {len(chunks)} chunks procesados")
                
            except Exception as e:
                print(f"Error al procesar {pdf_file}: {e}")
                continue

        if not all_chunks:
            print("⚠️ No se procesaron chunks")
            return

        try:
            # Generar puntos para Qdrant
            all_embeddings = self.generate_embeddings(all_chunks)
            
            points = [
                PointStruct(
                    id=global_id,
                    vector=embedding.tolist(),
                    payload={
                        "pdf_name": meta["pdf_name"],
                        "pdf_path": meta["pdf_path"],
                        "chunk_id": meta["chunk_id"],
                        "text": chunk
                    }
                )
                for global_id, (meta, embedding, chunk) in enumerate(zip(pdf_metadata, all_embeddings, all_chunks))
            ]

            # Almacenar en Qdrant
            await self.store_in_qdrant(points)

            # Guardar metadatos en JSON
            metadata_dict = {}
            for point in points:
                metadata_dict[point.id] = {
                    "pdf": point.payload["pdf_name"],
                    "pdf_path": point.payload["pdf_path"],
                    "chunk_id": point.payload["chunk_id"],
                    "chunk": point.payload["text"]
                }
            
            with open("pdf_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, ensure_ascii=False, indent=4)
            print("✅ Metadatos guardados en 'pdf_metadata.json'")
            
        except Exception as e:
            print(f"Error al procesar y almacenar PDFs: {e}")

    async def search_documents(self, query, top_k=5):
        """Realizar búsqueda en Qdrant"""
        if not self.model or not self.tokenizer:
            logger.warning("Modelo de embeddings no disponible, usando respuesta genérica")
            return [{"pdf": "Sistema", "texto": "Modelo de embeddings no disponible para búsqueda", "similitud": 0}]
            
        try:
            client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

            # Verificar conexión
            try:
                collection_info = await client.get_collection(self.collection_name)
                print(f"✅ Conexión a Qdrant exitosa")
            except Exception as e:
                print(f"❌ Error al conectar con Qdrant: {str(e)}")
                return []

            # Generar embedding de la consulta
            inputs = self.tokenizer([query], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

            # Buscar en Qdrant
            results = await client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )

            # Formatear resultados
            formatted_results = []
            for result in results:
                payload = result.payload or {}
                formatted_results.append({
                    "pdf": payload.get("pdf_name", "PDF desconocido"),
                    "texto": payload.get("text", "Texto no disponible"),
                    "similitud": round(result.score, 4)
                })

            return formatted_results
            
        except Exception as e:
            error_msg = f"Error en la búsqueda: {str(e)}"
            print(f"❌ {error_msg}")
            return [{"pdf": "Error", "texto": error_msg, "similitud": 0}]

    async def generar_respuesta_final(self, consulta_usuario, resultados, clasificacion, contexto_memoria=""):
        """Generar respuesta final usando el LLM unificado"""
        try:
            # Crear contexto con resultados de búsqueda
            contexto_busqueda = ""
            for i, res in enumerate(resultados, 1):
                contexto_busqueda += f"\n--- Fragmento {i} (PDF: {res['pdf']}) ---\n{res['texto']}\n"

            system_prompt = """Eres un profesor experto en física que proporciona explicaciones claras, precisas y didácticas.
Utiliza EXCLUSIVAMENTE el contexto proporcionado de los documentos para responder a la consulta del usuario.
Si la información en el contexto no es suficiente, responde: 'No tengo información suficiente en mis documentos para responder esa consulta.'
No utilices conocimiento general ni inventes información.
Estructura tu respuesta de manera clara, utilizando ecuaciones cuando sea apropiado y explicando los conceptos paso a paso.
IMPORTANTE: Nunca digas frases como 'Como modelo de lenguaje, no tengo memoria'. Actúa como si tuvieras memoria perfecta de la conversación y responde en consecuencia utilizando el contexto proporcionado.
Si en el contexto de conversación anterior hay alguna referencia relevante, úsala para dar continuidad a la conversación actual."""

            user_prompt = f"""
            CONSULTA DEL USUARIO: {consulta_usuario}

            CONTEXTO DE CONVERSACIÓN ANTERIOR:
            {contexto_memoria}

            CLASIFICACIÓN TEMÁTICA:
            {clasificacion}

            CONTEXTO DE DOCUMENTOS RELEVANTES:
            {contexto_busqueda}

            Por favor, proporciona una respuesta completa y didáctica a la consulta, basándote únicamente en la información proporcionada y manteniendo continuidad con conversaciones previas.
            """

            # Generar respuesta con el LLM unificado
            respuesta = await asyncio.to_thread(
                lambda: self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]).content
            )

            return respuesta
        except Exception as e:
            return f"Error al generar la respuesta final: {str(e)}"

    # Clase SemanticMemory interna
    class SemanticMemory:
        """Memoria semántica mejorada para mantener contexto de conversaciones"""
        
        def __init__(self, llm, max_tokens=2000):
            self.llm = llm
            self.max_tokens = max_tokens
            self.interactions = []
            self.summary = ""
            
        def add_interaction(self, user_query, ai_response):
            """Agregar una nueva interacción"""
            self.interactions.append({
                "timestamp": time.time(),
                "user": user_query,
                "ai": ai_response
            })
            
            # Mantener solo las últimas 5 interacciones para evitar sobrecarga
            if len(self.interactions) > 5:
                # Crear resumen de las interacciones más antiguas
                self._update_summary()
                self.interactions = self.interactions[-3:]  # Mantener las 3 más recientes
                
        def _update_summary(self):
            """Actualizar el resumen de conversaciones anteriores"""
            if len(self.interactions) <= 3:
                return
                
            # Crear texto de las interacciones más antiguas
            old_interactions = self.interactions[:-3]
            conversation_text = ""
            
            for interaction in old_interactions:
                conversation_text += f"Usuario: {interaction['user']}\n"
                conversation_text += f"AI: {interaction['ai'][:200]}...\n\n"
            
            # Generar resumen usando LLM
            try:
                summary_prompt = f"""
                Crea un resumen conciso de las siguientes interacciones de una conversación sobre física:
                
                {conversation_text}
                
                Resumen debe incluir:
                - Temas principales discutidos
                - Conceptos clave mencionados
                - Contexto importante para futuras consultas
                
                Mantén el resumen breve (máximo 300 palabras).
                """
                
                messages = [
                    SystemMessage(content="Eres un asistente que crea resúmenes concisos de conversaciones sobre física."),
                    HumanMessage(content=summary_prompt)
                ]
                
                response = self.llm.invoke(messages)
                
                # Combinar con resumen anterior si existe
                if self.summary:
                    self.summary = f"{self.summary}\n\n--- Resumen actualizado ---\n{response.content}"
                else:
                    self.summary = response.content
                    
            except Exception as e:
                print(f"Error al generar resumen: {e}")

        def get_context(self, max_length=1500):
            """Obtener contexto relevante para la conversación actual"""
            context_parts = []
            
            # Agregar resumen si existe
            if self.summary:
                context_parts.append(f"RESUMEN DE CONVERSACIONES ANTERIORES:\n{self.summary}")
            
            # Agregar interacciones recientes
            if self.interactions:
                recent_interactions = ""
                for interaction in self.interactions[-3:]:  # Últimas 3 interacciones
                    recent_interactions += f"\nUsuario: {interaction['user']}"
                    recent_interactions += f"\nAI: {interaction['ai'][:300]}...\n"
                
                if recent_interactions:
                    context_parts.append(f"INTERACCIONES RECIENTES:{recent_interactions}")
            
            # Combinar y limitar longitud
            full_context = "\n\n".join(context_parts)
            
            if len(full_context) > max_length:
                # Truncar manteniendo el final (más reciente)
                full_context = "..." + full_context[-max_length:]
            
            return full_context

        def clear_memory(self):
            """Limpiar la memoria de la conversación"""
            self.interactions = []
            self.summary = ""
            
        def get_conversation_stats(self):
            """Obtener estadísticas de la conversación"""
            return {
                "total_interactions": len(self.interactions),
                "has_summary": bool(self.summary),
                "memory_size": len(str(self.interactions) + self.summary)
            }

        # Métodos adicionales para PhysicsAgent (fuera de la clase SemanticMemory)

        def get_system_status(self):
            """Obtener estado del sistema"""
            status = {
                "pdfs_loaded": len(self.pdf_files),
                "pdfs_processed": self.pdfs_processed,
                "has_embedding_model": bool(self.model and self.tokenizer),
                "has_temario": bool(self.temario),
                "active_sessions": len(self.sessions),
                "qdrant_configured": bool(self.qdrant_url and self.qdrant_api_key)
            }
            return status

        def get_available_pdfs(self):
            """Obtener lista de PDFs disponibles"""
            return [os.path.basename(pdf) for pdf in self.pdf_files]

        def clear_session(self, session_id: str):
            """Limpiar una sesión específica"""
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False

        def clear_all_sessions(self):
            """Limpiar todas las sesiones"""
            self.sessions.clear()
            return True

        async def test_qdrant_connection(self):
            """Probar conexión con Qdrant"""
            try:
                client = AsyncQdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
                collections = await client.get_collections()
                return {
                    "connected": True,
                    "collections": [col.name for col in collections.collections],
                    "target_collection_exists": self.collection_name in [col.name for col in collections.collections]
                }
            except Exception as e:
                return {
                    "connected": False,
                    "error": str(e)
                }

        def get_temario_summary(self):
            """Obtener resumen del temario"""
            if not self.temario:
                return "Temario no disponible"
            
            # Truncar si es muy largo
            if len(self.temario) > 1000:
                return self.temario[:1000] + "..."
            return self.temario

        async def reprocess_pdfs(self):
            """Reprocesar todos los PDFs"""
            try:
                if not self.pdf_files:
                    return "No hay PDFs para procesar"
                
                print("🔄 Reprocesando PDFs...")
                
                # Procesar temario
                self.procesar_pdfs_temario(self.pdf_files)
                
                # Procesar y almacenar en Qdrant
                await self.procesar_y_almacenar_pdfs(self.pdf_files)
                
                self.pdfs_processed = True
                return "✅ PDFs reprocesados correctamente"
                
            except Exception as e:
                error_msg = f"Error al reprocesar PDFs: {str(e)}"
                logger.error(error_msg)
                return error_msg

        def search_in_temario(self, query: str):
            """Buscar en el temario directamente"""
            if not self.temario:
                return "Temario no disponible"
            
            # Búsqueda simple por palabras clave
            query_lower = query.lower()
            temario_lower = self.temario.lower()
            
            if query_lower in temario_lower:
                # Encontrar contexto alrededor de la coincidencia
                index = temario_lower.find(query_lower)
                start = max(0, index - 200)
                end = min(len(self.temario), index + 200)
                context = self.temario[start:end]
                
                return f"Encontrado en temario:\n...{context}..."
            
            return "No se encontró la consulta en el temario"

        def get_session_info(self, session_id: str):
            """Obtener información de una sesión específica"""
            if session_id not in self.sessions:
                return {"exists": False}
            
            session = self.sessions[session_id]
            return {
                "exists": True,
                "stats": session.get_conversation_stats(),
                "context_length": len(session.get_context())
            }

        def export_session_history(self, session_id: str):
            """Exportar historial de una sesión"""
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            return {
                "session_id": session_id,
                "interactions": session.interactions,
                "summary": session.summary,
                "exported_at": time.time()
            }

        # Método para agregar soporte de tipos de contenido
        def get_supported_content_types(self):
            """Obtener tipos de contenido soportados"""
            return self.SUPPORTED_CONTENT_TYPES

        def can_handle_content_type(self, content_type: str):
            """Verificar si puede manejar un tipo de contenido"""
            return content_type in self.SUPPORTED_CONTENT_TYPES