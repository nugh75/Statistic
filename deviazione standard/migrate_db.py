import sqlite3
from pathlib import Path

def migrate_database():
    # Percorso del database
    db_path = Path('instance/calcoli.db')
    
    if not db_path.exists():
        print("Database non trovato. Verrà creato un nuovo database.")
        return
    
    print("Inizio migrazione database...")
    
    # Connessione al database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Verifica se la colonna statistiche esiste già
        cursor.execute("PRAGMA table_info(calcolo)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'statistiche' not in columns:
            print("Aggiunta colonna 'statistiche'...")
            cursor.execute("ALTER TABLE calcolo ADD COLUMN statistiche TEXT")
            conn.commit()
            print("Colonna 'statistiche' aggiunta con successo.")
        else:
            print("La colonna 'statistiche' esiste già.")
        
        print("Migrazione completata con successo!")
    
    except Exception as e:
        print(f"Errore durante la migrazione: {str(e)}")
        conn.rollback()
    
    finally:
        conn.close()

if __name__ == '__main__':
    migrate_database()