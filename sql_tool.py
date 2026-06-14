import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

load_dotenv()

#1 Connexion a la db
db = SQLDatabase.from_uri("sqlite:///nba_data.db")

#2 Configruation du LLM (On pointe vers Mistral avec une température à 0 pour être précis)
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "mistral-large-latest"),
    api_key=os.getenv("MISTRAL_API_KEY"),
    base_url="https://api.mistral.ai/v1",
    temperature=0 
)

#3 Création du Prompt avec les exemples Few-Shot
#On explique à l'IA comment joindre les tables "players" et "stats"
template_personnalise = """
Tu es un expert en base de données SQLite pour la NBA. 
À partir de la question de l'utilisateur, crée UNIQUEMENT une requête SQL syntaxiquement correcte pour y répondre.
Ne renvoie rien d'autre que la requête SQL brute. Ne mets pas de balises ```sql.
Si aucune limite n'est précisée dans la question, limite ta requête à {top_k} résultats maximum.

Voici des exemples de questions et de requêtes correspondantes (Few-Shot) :

Question : "Combien de points a marqué Shai Gilgeous-Alexander ?"
SQLQuery : SELECT s.points FROM stats s JOIN players p ON s.player_id = p.player_id WHERE p.name = 'Shai Gilgeous-Alexander';

Question : "Quels sont les 5 joueurs ayant fait le plus de passes décisives (assists) ?"
SQLQuery : SELECT p.name, s.assists FROM stats s JOIN players p ON s.player_id = p.player_id ORDER BY s.assists DESC LIMIT 5;

Question : "Donne-moi le temps de jeu (minutes) moyen des joueurs de l'équipe OKC."
SQLQuery : SELECT AVG(s.minutes) FROM stats s JOIN players p ON s.player_id = p.player_id WHERE p.team = 'OKC';

Seules les tables suivantes sont disponibles :
{table_info}

Question : {input}
SQLQuery : 
"""

prompt = PromptTemplate.from_template(template_personnalise)

#4 Création de la chaîne et de l'outil d'exécution
#generate_query va écrire le SQL, execute_query va le lancer dans SQLite
generate_query = create_sql_query_chain(llm, db, prompt)
execute_query = QuerySQLDataBaseTool(db=db)

def interroger_base_sql(question: str) -> str:
    """
        Génère dynamiquement la requête SQL, l'exécute sur SQLite et retourne le résultat brut.
    """
    try:
        #Étape A : L'IA génère la requête SQL
        requete_sql = generate_query.invoke({"question": question})
        
        #Nettoyage de sécurité (au cas où Mistral ajoute du un truc pas net)
        requete_sql = requete_sql.replace("```sql", "").replace("```", "").strip()
        print(f"\n[LangChain Tool] Requête générée : {requete_sql}")
        
        #Étape B : LangChain exécute la requête sur la base
        resultat_brut = execute_query.invoke(requete_sql)
        
        #On retourne le résultat (généralement sous forme de liste de tuples)
        return f"Données brutes extraites de la base : {resultat_brut}"
        
    except Exception as e:
        return f"Erreur lors de l'exécution du Tool SQL : {e}"
    
# TEST INDÉPENDANT DU SCRIPT
if __name__ == "__main__":
    question_test = "Qui est le joueur qui a marqué le plus de points et combien en a-t-il marqué ?"
    print("Test de la question :", question_test)
    response = interroger_base_sql(question_test)
    print("->", response)