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
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file, abort, session
from models import db, Calcolo
from statistiche import StatisticheCalcolatore
import logging
from docx import Document
from io import BytesIO
from datetime import timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

# Initialize Flask app with explicit static folder configuration
app = Flask(__name__, 
    static_url_path='/static',
    static_folder='static')

# Get the absolute path for the database file
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'instance', 'calcoli.db')

# Configurazione avanzata
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev_key_for_session_management'),
    SQLALCHEMY_DATABASE_URI=f'sqlite:///{db_path}',
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SEND_FILE_MAX_AGE_DEFAULT=0,  # Disable cache for development
    DEBUG=False,  # Default to False for security
    TEMPLATES_AUTO_RELOAD=True,
    SESSION_COOKIE_SECURE=True,  # Only send cookie over HTTPS
    SESSION_COOKIE_HTTPONLY=True,  # Prevent JavaScript access to session cookie
    SESSION_COOKIE_SAMESITE='Lax',  # Protect against CSRF
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),  # Aumenta durata sessione a 30 minuti
    SESSION_TYPE='filesystem'  # Usa filesystem invece di memoria
)

# Aumenta la durata della sessione a 60 minuti e configura il salvataggio su filesystem
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev_key_for_session_management'),
    SQLALCHEMY_DATABASE_URI=f'sqlite:///{db_path}',
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SEND_FILE_MAX_AGE_DEFAULT=0,
    DEBUG=False,
    TEMPLATES_AUTO_RELOAD=True,
    SESSION_COOKIE_SECURE=False,  # Cambiato a False per supportare HTTP
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=60),
    SESSION_TYPE='filesystem',
    SESSION_FILE_DIR=tempfile.gettempdir()  # Directory temporanea per i file di sessione
)

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

# Add template filters
@app.template_filter('format_float')
def format_float(value):
    """Template filter per formattare numeri float."""
    try:
        return "{:.4f}".format(float(value))
    except (ValueError, TypeError):
        return str(value)

# Initialize database with app context
with app.app_context():
    db.init_app(app)
    db.create_all()

# Setup session clearing using before_request instead of before_first_request
@app.before_request
def clear_session_if_needed():
    if not hasattr(app, '_session_cleared'):
        session.clear()
        app._session_cleared = True

# Configura la sessione come permanente per ogni richiesta
@app.before_request
def make_session_permanent():
    session.permanent = True
    # Estendi la durata della sessione ad ogni richiesta
    session.modified = True

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
        # Gestione del caricamento iniziale del file
        if 'file' in request.files:
            try:
                file = request.files['file']
                nome = request.form.get('nome', 'Calcolo senza nome')
                note = request.form.get('note', '')
                
                if file.filename == '':
                    flash("Nessun file selezionato.")
                    return redirect(request.url)
                
                if not (file.filename.endswith('.xls') or file.filename.endswith('.xlsx')):
                    flash("Per favore carica un file Excel (.xls o .xlsx)")
                    return redirect(request.url)
                
                df = pd.read_excel(file)
                if df.empty:
                    flash("Il file Excel è vuoto")
                    return redirect(request.url)

                # Log before cleaning
                logging.info(f"Colonne originali: {df.columns.tolist()}")
                
                # Pulisci i nomi delle colonne rimuovendo caratteri speciali e spazi extra
                df.columns = [col.strip().replace('\r', ' ').replace('\n', ' ').strip() for col in df.columns]
                df.columns = [' '.join(col.split()) for col in df.columns]  # Rimuove spazi multipli
                
                # Log after cleaning
                logging.info(f"Colonne dopo pulizia: {df.columns.tolist()}")
                
                empty_cols = df.columns[df.isna().all()].tolist()
                if empty_cols:
                    flash(f"Le seguenti colonne sono vuote: {', '.join(empty_cols)}")
                
                # Store DataFrame in session
                session['temp_data'] = {
                    'columns': df.columns.tolist(),
                    'nome': nome,
                    'note': note,
                    'filename': file.filename
                }
                session['temp_df'] = df.to_json()
                session.modified = True
                
                logging.info(f"File caricato: {file.filename}, Colonne pulite: {df.columns.tolist()}")
                
                return render_template('select_series.html',
                                    columns=df.columns.tolist(),
                                    nome=nome,
                                    note=note)
                                    
            except Exception as e:
                logging.error(f"Errore durante il caricamento del file: {str(e)}")
                flash(f"Errore durante la lettura del file: {str(e)}")
                return redirect(request.url)
        
        # Controllo specifico per il form di selezione serie
        if request.form.get('step') == 'process_selection':
            logging.info(f"Form data ricevuta: {dict(request.form)}")
            
            try:
                # Recupera i dati dalla sessione
                df = pd.read_json(session.get('temp_df', '{}'))
                temp_data = session.get('temp_data', {})
                
                # Log column names from session
                logging.info(f"Colonne nel DataFrame dalla sessione: {df.columns.tolist()}")
                
                if df.empty or not temp_data:
                    logging.error("Dati della sessione mancanti o invalidi")
                    flash("Dati non validi o sessione scaduta. Ricarica il file.")
                    return redirect(url_for('index'))
                
                nome = temp_data.get('nome', 'Calcolo senza nome')
                note = temp_data.get('note', '')
                selected_series = request.form.getlist('selected_series')
                logging.info(f"Serie selezionate prima della pulizia: {selected_series}")
                
                if not selected_series:
                    logging.error("Nessuna serie selezionata")
                    flash("Seleziona almeno una serie da analizzare.")
                    return redirect(url_for('index'))
                
                # Pulisci i nomi delle colonne nel DataFrame come fatto durante il caricamento
                df.columns = [col.strip().replace('\r', ' ').replace('\n', ' ').strip() for col in df.columns]
                df.columns = [' '.join(col.split()) for col in df.columns]
                
                # Clean selected series names to match cleaned column names
                selected_series = [' '.join(serie.strip().replace('\r', ' ').replace('\n', ' ').split()) for serie in selected_series]
                logging.info(f"Serie selezionate dopo la pulizia: {selected_series}")
                logging.info(f"Colonne disponibili nel DataFrame: {df.columns.tolist()}")
                
                # Verifica corrispondenza
                for serie in selected_series:
                    if serie not in df.columns:
                        logging.error(f"Serie '{serie}' non trovata nelle colonne disponibili")
                        flash(f"Serie '{serie}' non trovata. Riprova la selezione.")
                        return redirect(url_for('index'))
                
                risultati = []
                all_series = {}
                
                # Process data and continue with existing logic...
                batch_size = 1000
                for colonna in selected_series:
                    logging.info(f"Processamento colonna: {colonna}")
                    if colonna not in df.columns:
                        logging.error(f"Colonna {colonna} non trovata nel DataFrame")
                        continue
                        
                    # Convert series to numeric, dropping non-numeric values
                    serie = pd.to_numeric(df[colonna], errors='coerce')
                    dati = serie.dropna().tolist()
                    
                    if not dati:
                        logging.warning(f"La serie '{colonna}' non contiene dati numerici validi.")
                        flash(f"La serie '{colonna}' non contiene dati numerici validi.")
                        continue
                        
                    if len(dati) > batch_size:
                        dati = dati[:batch_size]
                        flash(f"La serie '{colonna}' è stata limitata a {batch_size} valori.")
                    
                    # Perform statistical calculations for each series
                    try:
                        logging.info(f"Calcolo statistiche per {colonna}")
                        statistiche = StatisticheCalcolatore.calcola_tutte_statistiche(dati)
                        # Generate plots
                        plots = generate_plots(dati, colonna)
                        statistiche['plots'] = plots
                        
                        # Save series data and statistics
                        all_series[colonna] = {
                            'dati': dati,
                            'statistiche': statistiche
                        }
                        logging.info(f"Statistiche calcolate con successo per {colonna}")
                    except Exception as e:
                        logging.error(f"Errore nel calcolo delle statistiche per {colonna}: {str(e)}")
                        flash(f"Errore nel calcolo delle statistiche per la serie '{colonna}': {str(e)}")
                        continue

                # Analisi statistica per le serie selezionate
                if len(all_series) > 1:
                    logging.info("Calcolo correlazioni e test statistici per serie multiple")
                    # Prima generiamo la matrice di correlazione
                    series_data = {name: serie_info['dati'] for name, serie_info in all_series.items()}
                    matrice_correlazione_img, legenda = generate_correlation_matrix(series_data)
                    
                    # Poi calcoliamo correlazioni e t-test
                    correlazioni = StatisticheCalcolatore.calcola_correlazioni(series_data)
                    t_tests = StatisticheCalcolatore.calcola_ttest_coppie(series_data)
                    
                    # Aggiorniamo le statistiche per ogni serie
                    for colonna, serie_info in all_series.items():
                        statistiche = serie_info['statistiche']
                        # Genera i plot includendo tutte le serie per confronto
                        plots = generate_plots(serie_info['dati'], colonna, series_data)
                        statistiche['plots'] = plots
                        statistiche['correlazioni'] = correlazioni.get(colonna, {})
                        statistiche['t_tests'] = t_tests.get(colonna, {})
                        
                        if matrice_correlazione_img:
                            statistiche['plots']['correlation'] = matrice_correlazione_img
                        statistiche['legenda'] = legenda
                        
                        # Aggiorniamo le statistiche nel dizionario
                        all_series[colonna]['statistiche'] = statistiche
                        logging.info(f"Generati grafici completi per {colonna} con t-test e effect size")
                
                # Salvataggio nel database
                saved_count = 0
                for colonna, serie_info in all_series.items():
                    try:
                        dati = serie_info['dati']
                        statistiche = serie_info['statistiche']
                        
                        # Serializza i dati
                        stats_json = json.dumps(statistiche)
                        valori_json = json.dumps(dati)
                        
                        # Crea il record nel database
                        calcolo = Calcolo(
                            nome=nome,
                            note=note,
                            serie_nome=colonna,
                            valori=valori_json,
                            statistiche=stats_json
                        )
                        db.session.add(calcolo)
                        saved_count += 1
                        logging.info(f"Calcolo per {colonna} aggiunto al database con t-test e correlazioni")
                        
                    except Exception as e:
                        logging.error(f"Errore nel salvataggio dei dati per {colonna}: {str(e)}")
                        flash(f"Errore nel salvataggio dei dati per la serie {colonna}")
                        continue
                
                try:
                    if saved_count > 0:
                        db.session.commit()
                        logging.info("Commit al database completato con successo")
                        flash(f"Salvati con successo {saved_count} calcoli", "success")
                        return redirect(url_for('registro'))
                    else:
                        logging.error("Nessun calcolo salvato nel database")
                        flash("Nessun calcolo è stato salvato. Verifica i dati e riprova.")
                        return redirect(url_for('index'))
                        
                except Exception as e:
                    db.session.rollback()
                    logging.error(f"Errore nel commit al database: {str(e)}")
                    flash("Errore nel salvataggio dei risultati nel database.")
                    return redirect(url_for('index'))
                    
            except Exception as e:
                logging.error(f"Errore generale durante l'elaborazione del file: {str(e)}")
                flash(f"Errore durante l'elaborazione del file: {str(e)}")
                return redirect(url_for('index'))
            finally:
                # Clear session data only after successful processing
                if 'temp_df' in session and 'temp_data' in session:
                    session.pop('temp_df', None)
                    session.pop('temp_data', None)
                    session.modified = True
    
    return render_template('index.html')

@app.route('/registro')
def registro():
    try:
        calcoli = Calcolo.query.order_by(Calcolo.data_creazione.desc()).all()
        
        for calcolo in calcoli:
            if calcolo.statistiche:
                try:
                    stats = json.loads(calcolo.statistiche)
                    if isinstance(stats, dict):
                        # Ensure all required fields are present and properly formatted
                        required_fields = {
                            'count': int,
                            'media': float,
                            'mediana': float,
                            'deviazione_standard_popolazione': float,
                            'deviazione_standard_campione': float,
                            'varianza_popolazione': float,
                            'varianza_campione': float,
                            'range': float,
                            'quartili': dict,
                            'min_max': dict,
                            'plots': dict
                        }
                        
                        for field, convert_type in required_fields.items():
                            if field in stats:
                                if field in ['quartili', 'min_max', 'plots']:
                                    continue  # Skip conversion for dictionaries
                                try:
                                    stats[field] = convert_type(stats[field])
                                except (TypeError, ValueError):
                                    stats[field] = None
                        calcolo.statistiche = stats
                    else:
                        calcolo.statistiche = {}
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logging.warning(f"Invalid statistics for calculation {calcolo.id}: {str(e)}")
                    calcolo.statistiche = {}
            else:
                calcolo.statistiche = {}
        
        return render_template('registro.html', calcoli=calcoli)
    except Exception as e:
        logging.error(f"Error in registro route: {str(e)}")
        return render_template('500.html'), 500

@app.route('/calcoli/sposta', methods=['POST'])
def sposta_calcoli():
    return jsonify({'error': 'Operazione non più supportata'}), 400

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
        logging.error(f"Errore durante l'esportazione del PDF per l'ID {id}: {str(e)}")
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

@app.route('/risultato/<int:id>')
def visualizza_risultato(id):
    # Reindirizza al registro con l'ID del calcolo come parametro
    return redirect(url_for('registro', selected_id=id))

@app.route('/modifica-gruppo/<nome_attuale>/<nuovo_nome>')
def modifica_gruppo(nome_attuale, nuovo_nome):
    try:
        # Aggiorna il nome del gruppo per tutte le analisi corrispondenti
        Calcolo.query.filter_by(nome=nome_attuale).update({'nome': nuovo_nome})
        db.session.commit()
        flash('Gruppo rinominato con successo', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Errore durante la modifica del gruppo: {str(e)}', 'error')
    return redirect(url_for('registro'))

@app.route('/elimina-gruppo/<nome>')
def elimina_gruppo(nome):
    try:
        # Elimina tutte le analisi del gruppo
        Calcolo.query.filter_by(nome=nome).delete()
        db.session.commit()
        flash('Gruppo eliminato con successo', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Errore durante l\'eliminazione del gruppo: {str(e)}', 'error')
    return redirect(url_for('registro'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

import os
import zipfile
from io import BytesIO
from datetime import datetime

@app.route('/preview-export')
def preview_export():
    selected_ids = session.get('selected_items', [])
    if not selected_ids:
        flash('Nessun elemento selezionato per l\'esportazione', 'warning')
        return redirect(url_for('registro'))
    
    try:
        # Convert all IDs to integers since they might be stored as strings in session
        selected_ids = [int(id) for id in selected_ids]
        calcoli = Calcolo.query.filter(Calcolo.id.in_(selected_ids)).all()
        
        if not calcoli:
            flash('Nessun calcolo trovato per l\'esportazione', 'warning')
            return redirect(url_for('registro'))
            
        return render_template('preview_export.html', calcoli=calcoli)
        
    except (ValueError, TypeError) as e:
        flash('Errore nella selezione dei calcoli', 'error')
        logging.error(f"Errore nella conversione degli ID: {str(e)}")
        return redirect(url_for('registro'))
    except Exception as e:
        flash('Si è verificato un errore imprevisto', 'error')
        logging.error(f"Errore imprevisto in preview_export: {str(e)}")
        return redirect(url_for('registro'))

@app.route('/download-export')
def download_export():
    if 'selected_items' not in session:
        flash('Nessun elemento selezionato per l\'esportazione', 'warning')
        return redirect(url_for('registro'))
    
    selected_ids = session.get('selected_items', [])
    calcoli = Calcolo.query.filter(Calcolo.id.in_(selected_ids)).all()
    
    if not calcoli:
        flash('Nessun calcolo trovato per l\'esportazione', 'warning')
        return redirect(url_for('registro'))
    
    # Crea un documento Word
    doc = Document()
    doc.add_heading('Report Statistico', 0)
    
    for calcolo in calcoli:
        # Aggiungi titolo del calcolo
        doc.add_heading(f'Calcolo: {calcolo.nome}', level=1)
        doc.add_paragraph(f'Data: {calcolo.data_creazione.strftime("%d/%m/%Y")}')
        
        # Aggiungi statistiche di base
        statistiche = json.loads(calcolo.statistiche) if calcolo.statistiche else {}
        doc.add_heading('Statistiche di Base', level=2)
        if statistiche:
            doc.add_paragraph(f'Media: {statistiche.get("media", "N/A"):.2f}')
            doc.add_paragraph(f'Mediana: {statistiche.get("mediana", "N/A"):.2f}')
            doc.add_paragraph(f'Deviazione Standard: {statistiche.get("deviazione_standard_popolazione", "N/A"):.2f}')
        
        # Aggiungi note se presenti
        if calcolo.note:
            doc.add_heading('Note', level=2)
            doc.add_paragraph(calcolo.note)
        
        # Aggiungi un separatore tra i calcoli
        doc.add_paragraph('_' * 50)
    
    # Salva il documento in memoria
    memfile = BytesIO()
    doc.save(memfile)
    memfile.seek(0)
    
    return send_file(
        memfile,
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        as_attachment=True,
        download_name='report_statistico.docx'
    )

@app.route('/toggle-selection/<int:calcolo_id>', methods=['POST'])
def toggle_selection(calcolo_id):
    if 'selected_items' not in session:
        session['selected_items'] = []
    
    selected_items = session.get('selected_items', [])
    
    try:
        if calcolo_id in selected_items:
            selected_items.remove(calcolo_id)
        else:
            selected_items.append(calcolo_id)
        
        session['selected_items'] = selected_items
        session.modified = True
        
        return jsonify({
            'success': True,
            'selected': calcolo_id in selected_items,
            'count': len(selected_items)
        })
    except Exception as e:
        logging.error(f"Errore nella gestione della selezione: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/export')
def export():
    items = request.args.get('items', '').split(',')
    notes = request.args.get('notes', '')
    export_name = request.args.get('filename', '')
    
    if not items or items[0] == '':
        return "Nessun elemento selezionato", 400
        
    try:
        # Crea un buffer per il file ZIP
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # Crea una cartella per le immagini
            images_folder = 'images/'
            css_folder = 'css/'
            
            # Genera CSS
            css_content = """
            body { font-family: 'Roboto', sans-serif; line-height: 1.6; color: #2c3e50; background: #f5f7fa; }
            .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
            .result-card { background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-top: 1rem; }
            .stats-section { background: #f8f9fa; padding: 1rem; border-radius: 6px; border: 1px solid #e9ecef; }
            .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1.5rem; margin-top: 1rem; }
            .plot-card { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; text-align: center; }
            .plot-card img { max-width: 100%; height: auto; border-radius: 4px; }
            h1, h2, h3 { color: #2c3e50; }
            .note-box { background: #fff8dc; padding: 1.2rem; border-radius: 8px; margin: 1.5rem 0; border-left: 4px solid #3498db; }
            """
            zf.writestr(f'{css_folder}style.css', css_content)

            # Genera HTML per ogni calcolo
            calcoli = []
            plots_by_calcolo = {}
            
            for item_id in items:
                try:
                    calcolo = Calcolo.query.get(int(item_id))
                    if calcolo:
                        calcoli.append(calcolo)
                        plots = {}
                        statistiche = json.loads(calcolo.statistiche)
                        
                        # Salva i grafici
                        if 'plots' in statistiche:
                            for plot_type, plot_data in statistiche['plots'].items():
                                if plot_data:  # Skip empty plots
                                    img_data = base64.b64decode(plot_data)
                                    img_filename = f'{images_folder}{calcolo.id}_{plot_type}.png'
                                    zf.writestr(img_filename, img_data)
                                    plots[plot_type] = img_filename
                            
                            plots_by_calcolo[calcolo.id] = plots
                except:
                    continue

            if not calcoli:
                return "Nessun calcolo valido trovato", 400

            # Crea index.html con tutti i calcoli
            current_datetime = datetime.now()
            index_html = render_template('export_template.html',
                                       calcoli=calcoli,
                                       plots_by_calcolo=plots_by_calcolo,
                                       notes=notes,
                                       date=current_datetime)
            zf.writestr('index.html', index_html)

        # Prepara il file per il download
        memory_file.seek(0)
        
        # Genera nome file con data e ora se non specificato
        if not export_name:
            export_name = f'export_{current_datetime.strftime("%Y%m%d_%H%M%S")}'
        
        # Assicurati che il nome finisca con .zip
        if not export_name.lower().endswith('.zip'):
            export_name += '.zip'
            
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=export_name
        )
    except Exception as e:
        logging.error(f"Errore durante l'esportazione: {str(e)}")
        return f"Errore durante l'esportazione: {str(e)}", 500

@app.route('/result/<int:id>')
def result(id):
    calcolo = db.session.get(Calcolo, id)
    if calcolo is None:
        return abort(404)
    
    risultati = [{
        'serie': calcolo.serie_nome,
        'statistiche': json.loads(calcolo.statistiche) if calcolo.statistiche else {}
    }]
    
    return render_template('result.html', 
                         risultati=risultati,
                         nome=calcolo.nome,
                         note=calcolo.note)

if __name__ == '__main__':
    # Get configuration from environment variables
    debug_mode = os.environ.get('FLASK_ENV', 'production').lower() == 'development'
    port = int(os.environ.get('FLASK_PORT', 5003))
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    
    # Additional security check for debug mode
    if debug_mode and os.environ.get('FLASK_ENV') == 'development':
        app.debug = True
    else:
        app.debug = False
    
    app.run(
        port=port,
        threaded=True,
        host=host
    )