import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import tempfile
import zipfile
from scipy import stats
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file, abort
from models import db, Calcolo
from statistiche import StatisticheCalcolatore

# Initialize Flask app with explicit static folder configuration
app = Flask(__name__, 
    static_url_path='/static',
    static_folder='static')

# Configuration
app.config['SECRET_KEY'] = 'sostituisci_con_una_chiave_segreta'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///calcoli.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable cache for development

# Add cache control headers for static files
@app.after_request
def add_header(response):
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

# Add template filter for JSON parsing
@app.template_filter('from_json')
def from_json(value):
    return json.loads(value) if value else None

# Initialize database
db.init_app(app)
with app.app_context():
    db.create_all()

def generate_plots(data, title, all_series=None):
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

def generate_correlation_matrix(all_series):
    """
    Genera la matrice di correlazione per tutte le serie.
    
    Args:
        all_series: Dizionario con nome serie come chiave e lista di valori come valore
        
    Returns:
        tuple: (Base64 encoding dell'immagine della matrice di correlazione, dizionario della legenda)
    """
    if len(all_series) > 1:
        plt.clf()
        
        # Calcola la matrice di correlazione con i p-values
        series_data = {name: values for name, values in all_series.items() if len(values) > 0}
        correlazioni = StatisticheCalcolatore.calcola_correlazioni(series_data)
        
        # Calcola le dimensioni ottimali in base al numero di variabili
        n_vars = len(series_data)
        figsize = (min(12, max(8, n_vars * 1.2)), min(8, max(6, n_vars * 1.2)))
        
        # Crea il file temporaneo per la heatmap
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Genera la heatmap con etichette brevi e dimensioni calcolate
            legenda = StatisticheCalcolatore.crea_heatmap_correlazione(
                correlazioni,
                tmp.name, 
                use_etichette_brevi=True,
                figsize=figsize
            )
            
            # Leggi l'immagine salvata
            with open(tmp.name, 'rb') as f:
                img_data = f.read()
            
            import os
            os.unlink(tmp.name)  # Rimuovi il file temporaneo
            
            return base64.b64encode(img_data).decode('utf-8'), legenda
            
    return None, {}

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
            all_series = {}
            
            # Prima passiamo attraverso le colonne per raccogliere tutti i dati validi
            for colonna in df.columns:
                try:
                    dati = pd.to_numeric(df[colonna], errors='coerce').dropna().tolist()
                    if dati:
                        all_series[colonna] = dati
                except Exception as e:
                    print(f"[DEBUG] Errore nella conversione della colonna {colonna}: {str(e)}")

            # Calcola la matrice di correlazione una sola volta
            matrice_correlazione_img = None
            correlazioni = None
            legenda = {}
            if len(all_series) > 1:
                matrice_correlazione_img, legenda = generate_correlation_matrix(all_series)
                correlazioni = StatisticheCalcolatore.calcola_correlazioni(all_series)

            # Ora processiamo ogni serie per le statistiche
            for colonna, dati in all_series.items():
                try:
                    # Calcola statistiche
                    statistiche = StatisticheCalcolatore.calcola_tutte_statistiche(dati)
                    
                    # Prepara il dizionario delle statistiche
                    stats_dict = {
                        'count': len(dati),
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
                    
                    # Genera i grafici individuali
                    plots = generate_plots(dati, colonna)
                    stats_dict['plots'] = plots
                    
                    # Aggiungi la matrice di correlazione al primo risultato
                    if matrice_correlazione_img and len(risultati) == 0:
                        stats_dict['plots']['correlation'] = matrice_correlazione_img
                        stats_dict['legenda'] = legenda
                    
                    # Aggiungi le correlazioni se disponibili
                    if correlazioni:
                        stats_dict['correlazioni'] = correlazioni[colonna]
                    
                    # Serializza i dati
                    stats_json = json.dumps(stats_dict)
                    valori_json = json.dumps(dati)
                    
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
                    print(f"[DEBUG] Errore nell'elaborazione della serie {colonna}: {str(e)}")
                    flash(f"Errore nell'elaborazione della serie {colonna}: {str(e)}")
            
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

@app.route('/esporta_pdf/<int:id>')
def esporta_pdf(id):
    calcolo = db.session.get(Calcolo, id)
    if calcolo is None:
        return abort(404)
    
    try:
        # Carica le statistiche e i dati
        statistiche = json.loads(calcolo.statistiche) if calcolo.statistiche else {}
        serie_dati = json.loads(calcolo.valori) if calcolo.valori else []
        
        # Crea una directory temporanea per i file PDF
        import tempfile, os
        with tempfile.TemporaryDirectory() as temp_dir:
            # Assicurati che le statistiche includano tutti i dati necessari
            if 'plots' not in statistiche:
                # Rigenera i grafici se mancano
                plots = generate_plots(serie_dati, calcolo.serie_nome)
                statistiche['plots'] = plots
            
            pdf_path = StatisticheCalcolatore.esporta_pdf(
                calcolo.nome,
                statistiche,
                {calcolo.serie_nome: serie_dati},
                temp_dir
            )
            
            # Leggi il PDF generato
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            from flask import send_file
            import io
            
            # Invia il PDF come risposta
            return send_file(
                io.BytesIO(pdf_data),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'analisi_{calcolo.id}.pdf'
            )
            
    except Exception as e:
        import traceback
        print(f"Errore durante l'esportazione del PDF: {str(e)}")
        print(traceback.format_exc())  # Stampa lo stack trace completo
        flash(f"Errore durante l'esportazione del PDF: {str(e)}")
        return redirect(url_for('registro'))

@app.route('/esporta_pdf_multiplo', methods=['POST'])
def esporta_pdf_multiplo():
    try:
        data = request.get_json()
        if not data or 'series' not in data:
            return jsonify({'error': 'Nessuna serie selezionata'}), 400
        
        series_ids = data['series']
        if not series_ids:
            return jsonify({'error': 'Lista serie vuota'}), 400
        
        # Recupera i calcoli nell'ordine specificato
        calcoli = []
        for id in series_ids:
            calcolo = db.session.get(Calcolo, id)
            if calcolo:
                calcoli.append(calcolo)
        
        if not calcoli:
            return jsonify({'error': 'Nessun calcolo trovato'}), 404
        
        # Prepara i dati per il PDF
        series_data = {}
        all_statistics = []
        
        for calcolo in calcoli:
            statistiche = json.loads(calcolo.statistiche) if calcolo.statistiche else {}
            serie_dati = json.loads(calcolo.valori) if calcolo.valori else []
            
            # Assicurati che le statistiche includano tutti i dati necessari
            if 'plots' not in statistiche:
                plots = generate_plots(serie_dati, calcolo.serie_nome)
                statistiche['plots'] = plots
            
            series_data[calcolo.serie_nome] = serie_dati
            statistiche['nome_calcolo'] = calcolo.nome
            statistiche['serie_nome'] = calcolo.serie_nome
            statistiche['note'] = calcolo.note
            all_statistics.append(statistiche)
        
        # Crea una directory temporanea per il PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            # Genera il PDF con tutte le serie nell'ordine specificato
            pdf_path = StatisticheCalcolatore.esporta_pdf_multiplo(
                "Analisi Multiple",
                all_statistics,
                series_data,
                temp_dir
            )
            
            # Leggi il PDF generato
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            return send_file(
                io.BytesIO(pdf_data),
                mimetype='application/pdf',
                as_attachment=True,
                download_name='analisi_multiple.pdf'
            )
            
    except Exception as e:
        import traceback
        print(f"Errore durante l'esportazione multipla del PDF: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/esporta_html_multiplo', methods=['POST'])
def esporta_html_multiplo():
    try:
        data = request.get_json()
        if not data or 'series' not in data:
            return jsonify({'error': 'Nessuna serie selezionata'}), 400
        
        series_ids = data['series']
        if not series_ids:
            return jsonify({'error': 'Lista serie vuota'}), 400
        
        # Recupera i calcoli nell'ordine specificato
        calcoli = []
        for id in series_ids:
            calcolo = db.session.get(Calcolo, id)
            if calcolo:
                calcoli.append(calcolo)
        
        if not calcoli:
            return jsonify({'error': 'Nessun calcolo trovato'}), 404
        
        # Prepara i dati per l'HTML
        series_data = {}
        all_statistics = []
        
        for calcolo in calcoli:
            statistiche = json.loads(calcolo.statistiche) if calcolo.statistiche else {}
            serie_dati = json.loads(calcolo.valori) if calcolo.valori else []
            
            # Assicurati che le statistiche includano tutti i dati necessari
            if 'plots' not in statistiche:
                plots = generate_plots(serie_dati, calcolo.serie_nome)
                statistiche['plots'] = plots
            
            series_data[calcolo.serie_nome] = serie_dati
            statistiche['nome_calcolo'] = calcolo.nome
            statistiche['serie_nome'] = calcolo.serie_nome
            statistiche['note'] = calcolo.note
            all_statistics.append(statistiche)
        
        # Crea una directory temporanea per l'HTML e le immagini
        with tempfile.TemporaryDirectory() as temp_dir:
            # Genera l'HTML con tutte le serie nell'ordine specificato
            html_path = StatisticheCalcolatore.esporta_html_multiplo(
                "Analisi Multiple",
                all_statistics,
                series_data,
                temp_dir
            )
            
            # Crea un file zip contenente l'HTML e le immagini
            zip_path = os.path.join(temp_dir, 'report.zip')
            report_dir = os.path.dirname(html_path)
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, dirs, files in os.walk(report_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, report_dir)
                        zipf.write(file_path, arc_name)
            
            # Invia il file zip
            return send_file(
                zip_path,
                mimetype='application/zip',
                as_attachment=True,
                download_name='report_analisi.zip'
            )
            
    except Exception as e:
        import traceback
        print(f"Errore durante l'esportazione HTML: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/modifica/<int:id>', methods=['GET', 'POST'])
def modifica_calcolo(id):
    calcolo = db.session.get(Calcolo, id)
    if calcolo is None:
        return abort(404)
    if request.method == 'POST':
        calcolo.nome = request.form['nome']
        calcolo.note = request.form['note']
        db.session.commit()
        return redirect(url_for('registro'))
    return render_template('modifica.html', calcolo=calcolo)

@app.route('/elimina/<int:id>')
def elimina_calcolo(id):
    calcolo = db.session.get(Calcolo, id)
    if calcolo is None:
        return abort(404)
    db.session.delete(calcolo)
    db.session.commit()
    return redirect(url_for('registro'))

@app.route('/elimina_multipli', methods=['POST'])
def elimina_multipli():
    try:
        data = request.get_json()
        if not data or 'ids' not in data:
            return jsonify({'error': 'Nessun ID fornito'}), 400
        
        ids = data['ids']
        if not ids:
            return jsonify({'error': 'Lista ID vuota'}), 400
        
        # Delete all calcoli with the given IDs
        Calcolo.query.filter(Calcolo.id.in_(ids)).delete(synchronize_session=False)
        db.session.commit()
        
        return jsonify({'success': True}), 200
    except Exception as e:
        db.session.rollback()
        print(f"Errore durante l'eliminazione multipla: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.config['DEBUG'] = True  # Enable debug mode
    app.config['TEMPLATES_AUTO_RELOAD'] = True  # Enable template auto-reload
    app.run(debug=True, port=5002, threaded=True, host='0.0.0.0')