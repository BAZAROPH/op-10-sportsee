import sqlite3
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

##1 Validation des données via pydantic
class PlayerSchema(BaseModel):
    player_id:  int #Identifiant unique du joueur
    name: str #Nom du joueur
    team: str #Équipe actuelle
    position: Optional[str] = "Inconnu" #Poste du joueur


class MatchSchema(BaseModel):
    match_id: int #Identifiant unique du match
    date: str #Date de la rencontre
    home_team: str #Équipe à domicile
    away_team: str #Équipe à l'extérieur
    home_score: int = Field(ge=0) #Équipe à l'extérieur et impossible d'avoir un score négatif
    away_score: int = Field(ge=0) #Scores finaux


class StatSchema(BaseModel):
    stat_id: int #Identifiant unique de la ligne de statistique
    match_id: Optional[int] = None #Clé étrangère du matche (Optionnelle car pas de match dans l'excel)
    player_id: int #Clé étrangère du joueur
    #Métriques du joueur pour le match en question
    points: int = Field(ge=0)
    assists: int = Field(ge=0)
    rebounds: int = Field(ge=0)
    minutes:  float = Field(ge=0.0)


class ReportSchema(BaseModel):
    report_id: int #Identifiant unique du rapport
    match_id: int #Clé étrangère du matche
    title: str #Titre du rapport
    content: str #Contenu du rapport



##2 Création de la base de données SQLite
def init_db(db_name="nba_data.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    #Activation des clés étrangères pour SQLite
    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.executescript("""
        DROP TABLE IF EXISTS reports;
        DROP TABLE IF EXISTS stats;
        DROP TABLE IF EXISTS matches;
        DROP TABLE IF EXISTS players;
                         
        CREATE TABLE players(
            player_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            team TEXT NOT NULL,
            position TEXT
        );
                         
        CREATE TABLE matches(
            match_id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_score INTEGER,
            away_score INTEGER
        );
                         
        CREATE TABLE stats(
            stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NULL,
            player_id INTEGER,
            points INTEGER,
            assists INTEGER,
            rebounds INTEGER,
            minutes REAL,
            FOREIGN KEY(match_id) REFERENCES matches(match_id),
            FOREIGN KEY(player_id) REFERENCES players(player_id)
        );
                         
        CREATE TABLE reports(
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            title TEXT,
            content TEXT,
            FOREIGN KEY(match_id) REFERENCES matches(match_id)
        )
    """)
    conn.commit()
    return conn


##3 Lecture de l'excel et insertion
def transform_and_load(excel_file: str):
    if not os.path.exists(excel_file):
        logging.error(f"Fichier '{excel_file}' introuvable.")
        return

    db_name = "nba_data.db"
    conn = init_db(db_name)

    try:
        logging.info(f"Lecture du fichier {excel_file}...")
        
        df_stats = pd.read_excel(excel_file, sheet_name='Données NBA', header=1)
        
        #Nettoyage préventif des virgules françaises
        for col in df_stats.columns:
            if df_stats[col].dtype == 'object':
                try:
                    if df_stats[col].str.contains(',').any():
                        df_stats[col] = df_stats[col].astype(str).str.replace(',', '.').astype(float)
                except:
                    pass

        valid_players = []
        valid_stats = []
        
        #Étape A: validation Pydantic ligne par ligne
        for index, row in df_stats.iterrows():
            player_id = index + 1 
            
            player_data = {
                "player_id": player_id,
                "name": row['Player'],
                "team": row['Team'],
                "position": "Inconnu"
            }
            
            stat_data = {
                "stat_id": player_id,
                "match_id": None, 
                "player_id": player_id,
                "points": int(row['PTS']),
                "assists": int(row['AST']),
                "rebounds": int(row['REB']),
                "minutes": float(row['Min'])
            }

            try:
                valid_players.append(PlayerSchema(**player_data).model_dump())
                valid_stats.append(StatSchema(**stat_data).model_dump())
            except ValidationError as e:
                logging.warning(f"Ligne ignorée dans (index {index}): {e}")

        #Étape B : Insertion SQL
        if valid_players:
            df_valid_players = pd.DataFrame(valid_players)
            df_valid_players.to_sql('players', conn, if_exists='append', index=False)
            logging.info(f"Validation réussie : {len(valid_players)} lignes insérées dans 'players'.")
        
        if valid_stats:
            df_valid_stats = pd.DataFrame(valid_stats)
            df_valid_stats.to_sql('stats', conn, if_exists='append', index=False)
            logging.info(f"Validation réussie : {len(valid_stats)} lignes insérées dans 'stats'.")

    except Exception as e:
        logging.error(f"Erreur globale : {e}")
    finally:
        conn.close()


##4 Lancement du script
if __name__ == "__main__":
    transform_and_load("./inputs/regular NBA.xlsx")