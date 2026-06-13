import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd

import logfire
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

#Importer les modules existants
from utils.config import (EMBEDDING_MODEL, EMBEDDING_MODEL, MISTRAL_API_KEY, MODEL_NAME,
    SEARCH_K)
from utils.vector_store import VectorStoreManager
from MistralChat import generer_reponse, SYSTEM_PROMPT
from mistralai.models.chat_completion import ChatMessage

#Configuration de logfire
#Tracer tout ce qui se passe
logfire.configure()
logfire.instrument_requests()

#Configurer les modèles juges de ragas
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
        Fonction qui fait passer les questions de test dans le pipeline RAG actuel.
    """

    manager = VectorStoreManager()
    anwsers = []
    contexts = []

    for q in questions:
        #Recherche des docs
        search_results = manager.search(q, k=SEARCH_K)

        #Extraction du texte des chunks trouvés
        retrieved_texts = [res["text"] for res in search_results]
        contexts.append(retrieved_texts)

        #Formatage du contexte et génération de la réponse
        context_str = "\n\n".join(retrieved_texts)
        final_prompt = SYSTEM_PROMPT.format(context_str=context_str, question=q)

        response = generer_reponse([ChatMessage(role="user", content=final_prompt)])
        anwsers.append(response)

    return anwsers, contexts


def main():
    #3 Définir le dataset de test
    test_questions = [
        "Quels deux joueurs des Magic sont décrits comme un « absolute dogs » et un jeune duo d'ailiers prometteur ?",
        "Quelle équipe un commentateur cite-t-il comme ayant été la plus impressionnante à ses yeux ?"
    ]

    ground_truths = [
        "Paolo Banchero et Franz Wagner.",
        "Indiana Pacers et Minnesota Timberwolves."
    ]

    #Générer les prédictions par le prototype
    print("Génération des réponses par le système RAG...")
    answers, contexts = generate_rag_outputs(test_questions)

    #Créattion du dict de données pour ragas
    data = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    #4 eVALUATION
    print("Début de l'évaluation RAGAS ...")
    result = evaluate(
        dataset=dataset,
        metrics= [
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

    #Conversion des résultats Ragas en DataFrame Pandas
    df = result.to_pandas()

    #Définir des colonnes contenant les notes
    colonnes_notes = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']

    #Calcul de la moyenne pour chaque colonne (Pandas ignore automatiquement les 'nan' dans son calcul)
    moyennes = df[colonnes_notes].mean()

    #Création d'une nouvelle ligne "Moyenne" formatée
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

    #Ajout de la ligne à la fin du tableau existant
    df = pd.concat([df, ligne_moyenne], ignore_index=True)

    #Exportater en CSV
    df.to_csv("evaluation_resultats_avant_sql.csv", index=False)
    print("Fichier evaluation_resultats_avant_sql.csv généré avec succès (avec les moyennes en bas) !")

if __name__ == "__main__":
    main()