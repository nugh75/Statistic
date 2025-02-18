import sqlite3
from pathlib import Path

def rimuovi_cartelle():
    # Assicurati che la cartella instance esista
    Path('instance').mkdir(exist_ok=True)
    
    # Connetti al database
    conn = sqlite3.connect('instance/calcoli.db')
    cur = conn.cursor()
    
    try:
        # Controlla le colonne esistenti
        cur.execute("PRAGMA table_info(calcoli)")
        columns = [col[1] for col in cur.fetchall()]
        
        # Crea la query dinamicamente in base alle colonne esistenti
        selected_columns = ['id', 'nome', 'data_creazione', 'risultato', 'note', 'serie_nome']
        if 'valori' in columns:
            selected_columns.append('valori')
        if 'serie_dati' in columns:  # nel caso il nome sia serie_dati invece di valori
            selected_columns.append('serie_dati')
        if 'statistiche' in columns:
            selected_columns.append('statistiche')
            
        # Crea la tabella temporanea con le colonne corrette
        columns_sql = ', '.join(selected_columns)
        cur.execute(f'CREATE TABLE calcoli_temp AS SELECT {columns_sql} FROM calcoli')
        cur.execute('DROP TABLE calcoli')
        cur.execute('ALTER TABLE calcoli_temp RENAME TO calcoli')
        
        # Rimuovi la tabella folders
        cur.execute('DROP TABLE IF EXISTS folders')
        
        conn.commit()
        print("Migrazione completata con successo!")
        
    except Exception as e:
        print(f"Errore durante la migrazione: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    rimuovi_cartelle()