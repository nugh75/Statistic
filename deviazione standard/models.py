from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Calcolo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(100), nullable=False, default='Calcolo senza nome')
    data_creazione = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    risultato = db.Column(db.Float, nullable=False)  # Manteniamo per retrocompatibilità
    note = db.Column(db.Text)
    serie_nome = db.Column(db.String(100))
    valori = db.Column(db.Text)  # Memorizza i valori come stringa JSON
    statistiche = db.Column(db.Text)  # Memorizza tutte le statistiche come JSON