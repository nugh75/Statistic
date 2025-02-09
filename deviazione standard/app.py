import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy import stats
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from models import db, Calcolo
from statistiche import StatisticheCalcolatore

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sostituisci_con_una_chiave_segreta'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///calcoli.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Add template filter for JSON parsing
@app.template_filter('from_json')
def from_json(value):
    return json.loads(value) if value else None

db.init_app(app)

with app.app_context():
    db.create_all()

def generate_plots(data, title):
    plots = {}
    
    # Reset any existing plots
    plt.clf()
    
    # 1. Histogram with KDE and normal distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")  # Using seaborn's whitegrid style
    sns.histplot(data=data, stat='density', kde=True, ax=ax)
    
    # Add normal distribution curve
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    mu, std = stats.norm.fit(data)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'r-', lw=2, label='Distribuzione Normale')
    ax.set_title(f'Istogramma con KDE e Distribuzione Normale - {title}')
    ax.legend()
    
    # Save to base64 and close figure
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    plots['histogram'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    
    # 2. Box plot
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, ax=ax)
    ax.set_title(f'Box Plot - {title}')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    plots['boxplot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    
    # 3. Q-Q Plot
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot - {title}')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    plots['qqplot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    
    return plots

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("Nessun file caricato.")
            return redirect(request.url)
        
        file = request.files['file']
        nome = request.form.get('nome', 'Calcolo senza nome')
        note = request.form.get('note', '')
        
        if file.filename == '':
            flash("Nessun file selezionato.")
            return redirect(request.url)
        
        if not (file.filename.endswith('.xls') or file.filename.endswith('.xlsx')):
            flash("Per favore carica un file Excel (.xls o .xlsx)")
            return redirect(request.url)
        
        try:
            print(f"[DEBUG] Elaborazione file: {file.filename}")
            df = pd.read_excel(file)
            print(f"[DEBUG] File caricato con successo")
            print(f"[DEBUG] Colonne trovate: {df.columns.tolist()}")
            
            if df.empty:
                flash("Il file Excel è vuoto")
                return redirect(request.url)
            
            risultati = []
            
            for colonna in df.columns:
                try:
                    dati = pd.to_numeric(df[colonna], errors='coerce').dropna().tolist()
                    
                    if not dati:
                        print(f"[DEBUG] Colonna {colonna} saltata: nessun dato numerico valido")
                        continue
                    
                    print(f"[DEBUG] Dati validi trovati: {len(dati)}")
                    
                    # Calcola statistiche
                    statistiche = StatisticheCalcolatore.calcola_tutte_statistiche(dati)
                    
                    # Prepara il dizionario delle statistiche
                    stats_dict = {
                        'media': float(statistiche['media']),
                        'mediana': float(statistiche['mediana']),
                        'moda': [float(statistiche['moda'])] if isinstance(statistiche['moda'], (int, float)) else [float(x) for x in statistiche['moda']],
                        'deviazione_standard_popolazione': float(statistiche['deviazione_standard_popolazione']),
                        'deviazione_standard_campione': float(statistiche['deviazione_standard_campione']),
                        'varianza_popolazione': float(statistiche['varianza_popolazione']),
                        'varianza_campione': float(statistiche['varianza_campione']),
                        'range': float(statistiche['range']),
                        'quartili': {
                            'Q1': float(statistiche['quartili']['Q1']),
                            'Q2': float(statistiche['quartili']['Q2']),
                            'Q3': float(statistiche['quartili']['Q3'])
                        },
                        'min_max': {
                            'min': float(statistiche['min_max']['min']),
                            'max': float(statistiche['min_max']['max'])
                        }
                    }
                    
                    # Genera i grafici
                    plots = generate_plots(dati, colonna)
                    stats_dict['plots'] = plots
                    
                    # Serializza i dati prima di salvarli
                    try:
                        stats_json = json.dumps(stats_dict)
                        valori_json = json.dumps([float(x) for x in dati])
                        
                        # Crea il record nel database
                        calcolo = Calcolo(
                            nome=nome,
                            risultato=float(statistiche['deviazione_standard_popolazione']),
                            note=note,
                            serie_nome=colonna,
                            valori=valori_json,
                            statistiche=stats_json
                        )
                        db.session.add(calcolo)
                        
                        # Aggiungi ai risultati per la visualizzazione
                        risultati.append({
                            'serie': colonna,
                            'statistiche': stats_dict
                        })
                        
                    except Exception as e:
                        print(f"[DEBUG] Errore nella serializzazione dei dati: {str(e)}")
                        raise
                    
                except Exception as e:
                    print(f"[DEBUG] Errore nell'elaborazione della colonna {colonna}: {str(e)}")
                    flash(f"Errore nell'elaborazione della colonna {colonna}: {str(e)}")
            
            if not risultati:
                flash("Nessun dato numerico valido trovato nel file.")
                return redirect(request.url)
            
            try:
                db.session.commit()
                return render_template('result.html', risultati=risultati, nome=nome, note=note)
            except Exception as e:
                db.session.rollback()
                print(f"[DEBUG] Errore nel salvataggio nel database: {str(e)}")
                flash("Errore nel salvataggio dei risultati nel database.")
                return redirect(request.url)
                
        except Exception as e:
            print(f"[DEBUG] Errore generale: {str(e)}")
            flash(f"Errore durante l'elaborazione del file: {str(e)}")
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/registro')
def registro():
    calcoli = Calcolo.query.order_by(Calcolo.data_creazione.desc()).all()
    for calcolo in calcoli:
        if calcolo.statistiche:
            try:
                calcolo.statistiche = json.loads(calcolo.statistiche)
            except json.JSONDecodeError:
                calcolo.statistiche = None
    return render_template('registro.html', calcoli=calcoli)

@app.route('/modifica/<int:id>', methods=['GET', 'POST'])
def modifica_calcolo(id):
    calcolo = Calcolo.query.get_or_404(id)
    if request.method == 'POST':
        calcolo.nome = request.form['nome']
        calcolo.note = request.form['note']
        db.session.commit()
        return redirect(url_for('registro'))
    return render_template('modifica.html', calcolo=calcolo)

@app.route('/elimina/<int:id>')
def elimina_calcolo(id):
    calcolo = Calcolo.query.get_or_404(id)
    db.session.delete(calcolo)
    db.session.commit()
    return redirect(url_for('registro'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)