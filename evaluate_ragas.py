import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Contournement d'un problème de latence réseau (Happy Eyeballs / IPv6 mal configuré) :
# Par défaut, Python tente de se connecter en IPv6 aux API externes (comme api.mistral.ai).
# Si le réseau local supporte mal l'IPv6, cela provoque un blocage (hang) de 10 secondes par requête.
# Ce patch surcharge la résolution DNS pour forcer l'utilisation de l'IPv4, rendant les appels instantanés.
import socket
orig_getaddrinfo = socket.getaddrinfo
def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = ipv4_only_getaddrinfo

import pandas as pd

import logfire
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Importations des modules locaux
from utils.config import EMBEDDING_MODEL, MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager
from MistralChat import generer_reponse

# Configuration logfire
logfire.configure()
logfire.instrument_requests()

# Configuration des modèles juges pour Ragas
ragas_llm = ChatMistralAI(
    model_name=MODEL_NAME,
    mistral_api_key=MISTRAL_API_KEY,
    max_retries=10
)
ragas_embeddings = MistralAIEmbeddings(
    model=EMBEDDING_MODEL,
    mistral_api_key=MISTRAL_API_KEY,
    max_retries=10
)

def generate_rag_outputs(questions):
    """
    Exécute le pipeline RAG sur un set de questions via l'Agent Pydantic.
    """
    manager = VectorStoreManager()
    answers = []
    contexts = [] # Ce sera notre "Super Contexte" (FAISS + SQL)

    for q in questions:
        # 1. Extraction du contexte vectoriel (FAISS)
        search_results = manager.search(q, k=SEARCH_K)
        retrieved_texts = [res["text"] for res in search_results]

        # Formatage des dépendances
        context_str = "\n\n".join(retrieved_texts)
        if not retrieved_texts:
            context_str = "Aucune information textuelle trouvée."

        # On ajoute "sql_results" au sac à dos pour capturer la data
        safe_deps = {
            "context_str": context_str,
            "sql_queries": [],
            "sql_results": [],
            "vector_store_manager": manager
        }

        # 2. Appel à l'Agent hybride
        response, _ = generer_reponse(q, safe_deps, [])
        answers.append(response)

        # 3. CRÉATION DU SUPER CONTEXTE POUR RAGAS
        context_complet = retrieved_texts.copy() # On met d'abord le FAISS
        
        # On ajoute chaque résultat SQL trouvé par l'agent
        for sql_res in safe_deps["sql_results"]:
            context_complet.append(f"Données extraites de la base de données SQL : {sql_res}")
            
        # Sécurité pour Ragas (il déteste les contextes totalement vides)
        if not context_complet:
            context_complet = ["Aucune information contextuelle ni donnée SQL n'a été trouvée pour cette question."]

        contexts.append(context_complet)

    return answers, contexts
def main():
    # Définition du dataset de test
    test_questions = [
        "Qui est le joueur qui a marqué le plus de points et combien en a-t-il marqué ?",
        "Qui est le second meilleur joueur qui a marqué le plus de points et combien en a-t-il marqué ?",
        "Quelle équipe un commentateur cite-t-il comme ayant été la plus impressionnante à ses yeux ?",
        "Quel joueur de l'équipe de Miami (MIA) a marqué le plus de points durant la saison régulière, et quel est son score ?",
        "Combien de rebonds au total a récupéré le joueur Domantas Sabonis de l'équipe de Sacramento (SAC) ?",
        "Quel est le joueur de l'équipe de Boston (BOS) qui a joué en moyenne plus de 35 minutes par match, et quel est son temps de jeu exact ?",
        "Combien de fois Reggie Miller a-t-il été sélectionné pour le All-Star Game d'après les discussions Reddit, et pourquoi était-ce si difficile à son époque ?",
        "Quelle est la particularité budgétaire concernant la luxury tax lors d'une finale entre le Thunder et les Pacers mentionnée sur Reddit ?",
        "Selon les commentaires Reddit, quelle franchise NBA n'a pas obtenu l'avantage du terrain (home court advantage) jusqu'aux finales de la NBA ?",

        "Combien de points Shai Gilgeous-Alexander a-t-il inscrits en saison régulière, et quel constat sur la luxury tax est partagé sur Reddit dans le cas d'une finale entre son équipe (le Thunder) et les Pacers ?"
    ]

    ground_truths = [
        "Shai Gilgeous-Alexander est le joueur qui a marqué le plus de points lors de la saison régulière, avec un total de 2485 points",
        "Le second meilleur marqueur de la saison régulière est Anthony Edwards, avec un total de 2180 points",
        "Indiana Pacers et Minnesota Timberwolves.",
        "Tyler Herro est le meilleur marqueur de l'équipe de Miami (MIA) avec un total de 1840 points.",
        "Domantas Sabonis a récupéré un total de 973 rebonds durant la saison régulière.",
        "Jayson Tatum est le seul joueur de Boston (BOS) dans cette catégorie, avec une moyenne de 36.4 minutes par match.",
        "Reggie Miller n'a été sélectionné que 5 fois comme All-Star. C'était particulièrement difficile à son époque à l'Est car l'une des places de guard était systématiquement réservée et prise d'office par Michael Jordan.",
        "Ce serait la première finale NBA depuis la mise en place de la luxury tax où aucune des deux équipes (ni OKC Thunder, ni Indiana Pacers) n'a eu à payer cette taxe (neither team was a taxpayer) pour la saison concernée.",
        "Cela ne s'est jamais produit dans l'histoire de la NBA (Never in the NBA / Hasn't happened yet). Ce cas de figure s'est produit en NHL (avec les Edmonton Oilers) mais pas en NBA, car les meilleures équipes obtiennent d'ordinaire les meilleurs classements",
        

        "Shai Gilgeous-Alexander a marqué 2485 points en saison régulière. Sur Reddit, un post (Smith) souligne que si le Thunder (son équipe) et les Pacers s'affrontent en finale NBA, ce sera la première finale depuis la mise en place de la luxury tax où aucun des deux finalistes n'est assujetti à cette taxe (neither team was a taxpayer) pour cette saison."
    ]

    # Génération des prédictions
    print("Génération des réponses par le système RAG hybride...")
    answers, contexts = generate_rag_outputs(test_questions)

    # Structuration du dict de données Ragas
    data = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # Évaluation des métriques
    print("Début de l'évaluation Ragas ...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    print("--- Résultats de l'évaluation ---")
    print(result)

    # Conversion en DataFrame et calcul des moyennes
    df = result.to_pandas()
    colonnes_notes = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    moyennes = df[colonnes_notes].mean()

    # Création de la ligne d'agrégation
    ligne_moyenne = pd.DataFrame([{
        'question': 'MOYENNE GLOBALE',
        'answer': '---',
        'contexts': '---',
        'ground_truth': '---',
        'context_precision': moyennes['context_precision'],
        'context_recall': moyennes['context_recall'],
        'faithfulness': moyennes['faithfulness'],
        'answer_relevancy': moyennes['answer_relevancy']
    }])

    # Concaténation et export
    df = pd.concat([df, ligne_moyenne], ignore_index=True)
    
    # Export CSV
    output_filename = "evaluation_resultats_apres_sql-2.csv"
    df.to_csv(output_filename, index=False)
    print(f"Fichier {output_filename} généré avec succès (avec les moyennes) !")

if __name__ == "__main__":
    main()