# Assistant RAG Hybride avec Mistral (NBA Analyst AI)

Ce projet implémente un assistant virtuel expert de la NBA (**NBA Analyst AI**) utilisant une architecture de **Retrieval-Augmented Generation (RAG) Hybride** basée sur les modèles Mistral. Il combine la recherche sémantique vectorielle (sur des documents non structurés) et le requêtage de base de données relationnelle (sur des statistiques structurées) pour fournir des réponses ultra-précises et contextualisées.

> [!NOTE]
> Pour une explication très détaillée de l'architecture interne, du fonctionnement du pipeline hybride et de l'évaluation scientifique des performances, consultez la [Documentation de la Méthodologie RAG (README_RAG.md)](./README_RAG.md).

---

## Fonctionnalités

- 🔍 **Recherche sémantique vectorielle** : Recherche par similarité cosinus avec FAISS dans les archives textuelles (PDFs de discussions Reddit) en utilisant le modèle `mistral-embed`.
- 📊 **Requêtage SQL Relationnel dynamique** : Génération et exécution de requêtes SQL SQLite via LangChain à partir des questions de l'utilisateur pour extraire les statistiques exactes des joueurs.
- 🤖 **Orchestration d'Agent Intelligent** : Agent orchestrateur basé sur le framework **Pydantic AI** (utilisant `mistral-small-latest`) qui choisit dynamiquement et séquentiellement d'interroger les documents (FAISS) ou la base de données (SQL).
- 👁️ **Fallback OCR automatique** : Extraction robuste des documents PDF avec basculement automatique sur un moteur OCR (**PyMuPDF + EasyOCR**) si les documents sont des scans.
- 📈 **Évaluation de performances Ragas & Logfire** : Évaluation intégrée du RAG avec des métriques standardisées de précision, rappel et fidélité de réponse, instrumentée avec Logfire.

---

## 📂 Structure du Projet

```
.
├── MistralChat.py           # Application Streamlit principale (UI & Agent Pydantic AI)
├── load_excel_to_db.py      # Validation Pydantic et importation du Excel regular NBA dans SQLite
├── indexer.py               # Extraction et indexation vectorielle des PDF dans FAISS
├── sql_tool.py              # Traducteur LangChain Langage Naturel -> Requête SQL SQLite
├── evaluate_ragas.py        # Évaluation automatique du système RAG avec Ragas et Logfire
├── compare_eval.py          # Comparatif et génération du graphique de performances (Avant vs Après SQL)
├── inputs/                  # Répertoire des sources (Fichiers PDF Reddit et regular NBA.xlsx)
├── vector_db/               # Index vectoriel FAISS et chunks persistés (pickle)
├── nba_data.db              # Base de données relationnelle SQLite contenant les statistiques NBA
├── requirements.txt         # Fichier des dépendances Python
├── utils/                   # Modules utilitaires partagés
│   ├── config.py            # Configuration globale de l'application (modèles, chunk_size, etc.)
│   ├── data_loader.py       # Chargement, parsing de fichiers et pipeline OCR
│   ├── schemas.py           # Schémas de validation des chunks de documents
│   └── vector_store.py      # Gestionnaire d'index vectoriel FAISS et appels embeddings
└── README_RAG.md            # Méthodologie et analyse comparative du RAG
```

---

## 🛠️ Prérequis

- Python 3.9+
- Clé API Mistral (à obtenir sur [console.mistral.ai](https://console.mistral.ai/))

---

## 📦 Installation

1. **Cloner le dépôt**
   ```bash
   git clone git@github.com:BAZAROPH/op-10-sportsee.git
   cd op-10-sportsee
   ```

2. **Créer un environnement virtuel**
   ```bash
   # Création
   python -m venv .venv

   # Activation
   # Sur Windows
   .venv\Scripts\activate
   # Sur macOS/Linux
   source .venv/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer les variables d'environnement**
   Créez un fichier `.env` à la racine du projet avec vos accès :
   ```env
   MISTRAL_API_KEY=votre_cle_api_mistral
   ```

---

## 📖 Utilisation

### 1. Ingestion et initialisation des données

Pour initialiser le système complet, exécutez séquentiellement ces deux scripts :

```bash
# Étape A : Valider et charger le fichier Excel des statistiques dans SQLite
python load_excel_to_db.py

# Étape B : Extraire, découper et indexer les PDFs dans la base vectorielle FAISS
python indexer.py
```

### 2. Lancer l'application web Streamlit

```bash
streamlit run MistralChat.py
```
L'application s'ouvre automatiquement dans votre navigateur à l'adresse `http://localhost:8501`. 
L'interface vous permet de converser avec **NBA Analyst AI** et de voir, grâce à des volets dépliants (expanders), quelles requêtes SQL ou recherches FAISS ont été exécutées en arrière-plan.

### 3. Lancer l'évaluation Ragas et visualiser les gains de performance

Vous pouvez évaluer la qualité des réponses et générer les courbes de comparaison :
```bash
# Génère les rapports d'évaluation Ragas au format CSV
python evaluate_ragas.py

# Génère le graphique comparatif avant/après intégration SQL
python compare_eval.py
```
Le graphique sera exporté sous le nom `comparatif_metrics_ragas.png`.

---

## ⚙️ Personnalisation

La configuration de l'application est centralisée dans le fichier [utils/config.py](./utils/config.py) :
- **Taille & Chevauchement des Chunks** : Ajustez `CHUNK_SIZE` et `CHUNK_OVERLAP` pour modifier le comportement de découpage textuel.
- **Modèles utilisés** : Remplacez `MODEL_NAME` (par défaut `mistral-small-latest`) ou `EMBEDDING_MODEL` (`mistral-embed`).
- **Nombre de documents retournés** : Modifiez `SEARCH_K` pour contrôler le nombre de chunks FAISS à injecter dans le contexte initial.
