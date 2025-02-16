from app import app, db

with app.app_context():
    print("Creazione delle tabelle del database...")
    db.create_all()
    print("Database inizializzato con successo!")