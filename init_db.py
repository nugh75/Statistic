from flask import Flask
from models import db
from datetime import datetime
import sqlite3
from pathlib import Path
import os
from file_manager import FileManager

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/calcoli.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def init_db():
    # Initialize storage structure
    file_manager = FileManager('storage')
    
    # Assicurati che la cartella instance esista e abbia i permessi corretti
    instance_path = Path('instance')
    instance_path.mkdir(exist_ok=True)
    os.chmod(instance_path, 0o755)
    
    db_path = instance_path / 'calcoli.db'
    if db_path.exists():
        os.chmod(db_path, 0o666)
    
    # Connettiti al database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Crea le tabelle
    cur.executescript('''
        -- Crea la tabella folders
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            parent_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_id) REFERENCES folders (id)
        );
        
        -- Crea la tabella calcoli con il nuovo campo folder_id
        CREATE TABLE IF NOT EXISTS calcoli (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            data_creazione TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            valori TEXT,
            statistiche TEXT,
            serie_nome TEXT,
            note TEXT,
            folder_id INTEGER REFERENCES folders(id)
        )
    ''')
    
    # Verifica se esiste già una cartella root
    cur.execute('SELECT id FROM folders WHERE name = "Root"')
    root = cur.fetchone()
    
    if not root:
        # Crea la cartella root
        cur.execute('''
            INSERT INTO folders (name, description)
            VALUES (?, ?)
        ''', ('Root', 'Cartella principale'))
        root_id = cur.lastrowid
        
        # Create physical root folder
        file_manager.create_folder(root_id, 'Root')
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()