# compare_eval.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##/ Configuration des chemins d'accès des rapports d'évaluation
file_avant = "evaluation_resultats_avant_sql-1.csv"
file_apres = "evaluation_resultats_apres_sql-2.csv"
#/

# 1. Chargement et vérification des fichiers d'extraction
try:
    df_avant = pd.read_csv(file_avant)
    df_apres = pd.read_csv(file_apres)
except FileNotFoundError as e:
    print(f"Erreur d'acquisition des données : {e}")
    exit(1)

##1 Extraction des lignes d'agrégation globale
# Isolation des métriques moyennes calculées en fin de rapport
data_avant = df_avant[df_avant['question'] == 'MOYENNE GLOBALE'].iloc[0]
data_apres = df_apres[df_apres['question'] == 'MOYENNE GLOBALE'].iloc[0]

# Mapping des métriques Ragas à représenter
metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
labels_metrics = ['Précision Contexte', 'Rappel Contexte', 'Fidélité (Faithfulness)', 'Pertinence Réponse']

scores_avant = [data_avant[m] for m in metrics]
scores_apres = [data_apres[m] for m in metrics]
#/

##2 Configuration de la géométrie du graphique
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

# Génération des barres juxtaposées (Avant vs Après)
rects_avant = ax.bar(x - width/2, scores_avant, width, label='Avant (RAG Documentaire seul)', color='#e056fd')
rects_apres = ax.bar(x + width/2, scores_apres, width, label='Après (RAG Hybride + SQL)', color='#22a6b3')
#/

##3 Traitement des annotations et habillage de l'axe
ax.set_ylabel('Valeur des Métriques (0.0 à 1.0)', fontsize=11)
ax.set_title('Impact de l\'intégration de la base SQL sur les performances Ragas', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels_metrics, fontsize=10)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)

def implanter_valeurs(rects):
    """
    Positionne les labels de scores au sommet de chaque barre géométrique.
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

implanter_valeurs(rects_avant)
implanter_valeurs(rects_apres)
#/

# 4. Exportation et rendu final du pipeline visuel
plt.tight_layout()
output_chart = "comparatif_metrics_ragas.png"
plt.savefig(output_chart, dpi=300)
print(f"Graphique comparatif exporté avec succès sous : {output_chart}")
plt.show()