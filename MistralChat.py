# MistralChat.py
# Contournement d'un problème de latence réseau (Happy Eyeballs / IPv6 mal configuré) :
# Par défaut, Python tente de se connecter en IPv6 aux API externes (comme api.mistral.ai).
# Si le réseau local supporte mal l'IPv6, cela provoque un blocage (hang) de 10 secondes par requête.
# Ce patch surcharge la résolution DNS pour forcer l'utilisation de l'IPv4, rendant les appels instantanés.
import socket
orig_getaddrinfo = socket.getaddrinfo
def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = ipv4_only_getaddrinfo

import streamlit as st
import os
import logging
from typing import Any
##/
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire
from sql_tool import interroger_base_sql
#/

logfire.configure()
logfire.instrument_pydantic()

# --- Configuration et imports des constantes applicatives ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME
    )
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()


# --- Initialisation du logger système ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Validation des variables d'environnement ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()

##/
#Initialisation du client de communication avec l'infrastructure Mistral
mistral_model = OpenAIChatModel(
    model, 
    provider=OpenAIProvider(
        base_url="https://api.mistral.ai/v1",
        api_key=api_key
    )
)
#Définition du gestionnaire principal et de ses directives métier
nba_agent = Agent(
    mistral_model,
    system_prompt=(
        f"Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball {NAME}. "
        "Ta mission est de répondre aux questions des fans en animant le débat. "
        "Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre."
    ),
    retries=3 # Nombre de tentatives d'exécution en cas d'erreur de parsing
)

#Injection dynamique du contexte de recherche dans la session courante
@nba_agent.system_prompt
def add_context_to_prompt(ctx: RunContext[dict]) -> str:
    return f"""\n\n--- CONTEXTE TROUVÉ ---\n{ctx.deps['context_str']}\n---
            Tu peux tout à fait utiliser tes différents outils séquentiellement pour une même question. 
            Si l'utilisateur demande des statistiques précises ET un contexte historique, appelle l'outil SQL puis l'outil FAISS avant de rédiger ta réponse finale.
            """

#Module d'extraction vectorielle (Index FAISS)
@nba_agent.tool
def recherche_documentaire_faiss(ctx: RunContext[Any], question_recherche: str) -> str:
    """
        Recherche par similarité sémantique au sein des archives non structurées (PDF).
    """
    logging.info(f"L'agent utilise l'outils FAISS pour : {question_recherche}")

    # Indique qu'on a appelé FAISS dans les dépendances (évite l'accès direct à st.session_state dans un thread séparé)
    if isinstance(ctx.deps, dict):
        ctx.deps["faiss_called"] = True

    # Récupération du VectorStoreManager
    if isinstance(ctx.deps, dict) and "vector_store_manager" in ctx.deps:
        manager = ctx.deps["vector_store_manager"]
    else:
        manager = ctx.deps

    if manager is None:
        return "Erreur: la base de documents n'est pas disponible."
    
    try:
        search_results = manager.search(question_recherche, k=SEARCH_K)
        if not search_results:
            return "Aucune information trouvée dans les documents pour cette question."
        context_str = "\n\n".join([f"Source Document: {res['text']}" for res in search_results])

        return f"Voici ce que disent les documents :\n{context_str}"
    
    except Exception as e:
        return f"Erreur lors de la recherche dans les documents : {e}"

#Module d'extraction relationnelle (Base SQL)
@nba_agent.tool
def recherche_statistiques_sql(ctx: RunContext[dict], requete_utilisateur: str) -> str:
    """
    Interroge la base de données SQL pour récupérer des statistiques précises sur les joueurs (points, passes, rebonds, minutes).
    Le paramètre 'requete_utilisateur' doit être une question en langage naturel (ex: 'Combien de points a marqué Shai Gilgeous-Alexander ?').
    """
    logging.info(f"L'Agent délègue au Tool SQL LangChain la requête : {requete_utilisateur}")
    
    reponse_outil = interroger_base_sql(requete_utilisateur) 

    #On écrit la requête dans les dépendances pour l'UI, et le résultat pour RAGAS
    if isinstance(reponse_outil, dict):
        ctx.deps["sql_queries"].append(reponse_outil.get("requete", "Requête non affichable"))
        
        #On sauvegarde la donnée brute si le capteur existe
        if "sql_results" in ctx.deps:
            ctx.deps["sql_results"].append(reponse_outil.get("resultat", ""))
            
        return reponse_outil.get("resultat", "")
    else:
        ctx.deps["sql_queries"].append(str(reponse_outil))
        if "sql_results" in ctx.deps:
            ctx.deps["sql_results"].append(str(reponse_outil))
        return str(reponse_outil)
#/

# --- Persistance et mise en cache de la structure vectorielle ---
@st.cache_resource
def get_vector_store_manager():
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            return None
        return manager
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement: {e}")
        return None

vector_store_manager = get_vector_store_manager()

# --- Modèle de prompt structurel ---
SYSTEM_PROMPT = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{{context_str}}
---

QUESTION DU FAN:
{{question}}

RÉPONSE DE L'ANALYSTE NBA:"""


# --- Instanciation et vérification des états de session Streamlit ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes."}]

#/
if "agent_history" not in st.session_state:
    #Structure de stockage de l'historique 
    st.session_state.agent_history = []
#/

#Structure de tracking pour l'affichage conditionnel des pipelines d'extraction
if "sources_en_cours" not in st.session_state:
    st.session_state.sources_en_cours = {"faiss": False, "sql": []}


def generer_reponse(prompt: str, deps_dict: dict, history: list):
    """
    Exécute le cycle d'interrogation synchrone avec injection du dictionnaire de dépendances.
    """
    try:
        logging.info(f"Appel à l'Agent Pydantic AI avec {len(history)} messages en mémoire.")

        result = nba_agent.run_sync(
            prompt,
            deps=deps_dict,
            message_history=history
        )
        return result.output, result.all_messages()

    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'agent pydantic AI: {e}")
        logging.exception("Erreur API Agent")
        # Rétablissement séquentiel de l'historique en cas d'échec de la transaction
        return "Je suis désolé, une erreur technique m'empêche de répondre.", history

# --- Construction de la couche de présentation (UI Streamlit) ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")

#Rendu de l'historique des échanges de la session
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        #Hydratation conditionnelle de l'arborescence des composants sources
        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources.get("faiss") or len(sources.get("sql", [])) > 0:
                with st.expander("🔍 Voir les sources et requêtes en arrière-plan"):
                    if sources.get("faiss"):
                        st.info("📚 Recherche documentaire effectuée dans l'index FAISS (PDFs).")
                    for sql_query in sources.get("sql", []):
                        st.success("💾 Recherche dans la base de données (SQL) effectuée.")
                        st.code(sql_query, language="sql")

#Logique de capture et traitement du flux d'entrée utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    #Réinitialisation du dictionnaire d'état pour la nouvelle transaction
    st.session_state.sources_en_cours = {"faiss": False, "sql": []}

    #Pipeline RAG d'extraction vectorielle amont
    if vector_store_manager is None:
        st.error("Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande.")
        logging.error("VectorStoreManager non disponible pour la recherche.")
        st.stop()

    try:
        logging.info(f"Recherche de contexte pour la question: '{prompt}' avec k={SEARCH_K}")
        search_results = vector_store_manager.search(prompt, k=SEARCH_K)
        logging.info(f"{len(search_results)} chunks trouvés dans le Vector Store.")
        
        if search_results:
            st.session_state.sources_en_cours["faiss"] = True
            
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la recherche d'informations pertinentes: {e}")
        logging.exception(f"Erreur pendant vector_store_manager.search pour la query: {prompt}")
        search_results = []

    # Formatage de la charge utile contextuelle textuelle
    context_str = "\n\n---\n\n".join([
        f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
        for res in search_results
    ])

    if not search_results:
        context_str = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
        logging.warning(f"Aucun contexte trouvé pour la query: {prompt}")

    #Section de rendu dynamique du flux de l'assistant
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") 

        #Initialisation du conteneur d'isolation pour le thread de requêtage SQL
        safe_deps = {
            "context_str": context_str,
            "sql_queries": [],
            "vector_store_manager": vector_store_manager
        }

        response_content, st.session_state.agent_history = generer_reponse(
            prompt, 
            safe_deps, 
            st.session_state.agent_history
        )

        #Rapprochement des requêtes collectées de façon asynchrone avec le contexte Streamlit
        st.session_state.sources_en_cours["sql"] = safe_deps["sql_queries"]
        if safe_deps.get("faiss_called"):
            st.session_state.sources_en_cours["faiss"] = True

        message_placeholder.write(response_content)
        
        #Injection immédiate du composant expander si des outils d'arrière-plan ont été sollicités
        sources_utilisees = st.session_state.sources_en_cours
        if sources_utilisees["faiss"] or len(sources_utilisees["sql"]) > 0:
            with st.expander("🔍 Voir les sources et requêtes en arrière-plan"):
                if sources_utilisees["faiss"]:
                    st.info("📚 Recherche documentaire effectuée dans l'index FAISS (PDFs).")
                for sql_query in sources_utilisees["sql"]:
                    st.success("💾 Recherche dans la base de données (SQL) effectuée.")
                    st.code(sql_query, language="sql")

    #Commisération immuable de la réponse et de ses métadonnées dans l'état de session historique
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_content,
        "sources": st.session_state.sources_en_cours.copy()
    })

#Rendu de l'identité visuelle de bas de page
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")