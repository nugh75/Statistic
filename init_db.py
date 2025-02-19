from flask import Flask
from models import db, Folder, Calcolo
import os

app = Flask(__name__)

# Get the absolute path for the database file
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'instance', 'calcoli.db')

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def init_db():
    with app.app_context():
        # Ensure instance directory exists
        instance_dir = os.path.join(basedir, 'instance')
        os.makedirs(instance_dir, exist_ok=True)
        
        # Remove existing database if present
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Database esistente rimosso: {db_path}")
        
        # Create all tables
        db.create_all()
        print("Tabelle create con successo")
        
        # Check if root folder exists
        root = Folder.query.filter_by(name="Root").first()
        
        if not root:
            # Create root folder
            root = Folder(
                name="Root",
                description="Cartella principale"
            )
            db.session.add(root)
            try:
                db.session.commit()
                print("Cartella root creata con successo")
            except Exception as e:
                print(f"Errore durante la creazione della cartella root: {e}")
                db.session.rollback()

if __name__ == '__main__':
    try:
        init_db()
        print("\nDatabase inizializzato correttamente!")
        
        # Verify table structure
        with app.app_context():
            # Check folders table
            folders = db.session.execute(db.text('PRAGMA table_info(folders)')).fetchall()
            print("\nStruttura tabella folders:")
            for col in folders:
                print(f"- {col[1]} ({col[2]})")
            
            # Check calcoli table
            calcoli = db.session.execute(db.text('PRAGMA table_info(calcoli)')).fetchall()
            print("\nStruttura tabella calcoli:")
            for col in calcoli:
                print(f"- {col[1]} ({col[2]})")
            
    except Exception as e:
        print(f"Errore durante l'inizializzazione del database: {e}")