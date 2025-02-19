from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import json

db = SQLAlchemy()

class Folder(db.Model):
    __tablename__ = 'folders'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    parent_id = db.Column(db.Integer, db.ForeignKey('folders.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relazione con i calcoli
    calcoli = db.relationship('Calcolo', backref='folder', lazy=True)

class Calcolo(db.Model):
    __tablename__ = 'calcoli'
    
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(200), nullable=False)
    data_creazione = db.Column(db.DateTime, default=datetime.utcnow)
    valori = db.Column(db.Text)  # JSON string dei valori
    statistiche = db.Column(db.Text)  # JSON string delle statistiche
    serie_nome = db.Column(db.String(100))
    note = db.Column(db.Text)
    folder_id = db.Column(db.Integer, db.ForeignKey('folders.id'))

class ExportSelection(db.Model):
    __tablename__ = 'export_selections'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), nullable=False)
    calcolo_id = db.Column(db.Integer, db.ForeignKey('calcoli.id'), nullable=False)
    note = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    calcolo = db.relationship('Calcolo', backref='selections')