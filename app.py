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

# Configuration
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev_key_for_session_management'),
    SQLALCHEMY_DATABASE_URI=f'sqlite:///{db_path}',
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SEND_FILE_MAX_AGE_DEFAULT=0,  # Disable cache for development
    DEBUG=False,  # Default to False for security
    TEMPLATES_AUTO_RELOAD=True,
    SESSION_COOKIE_SECURE=True,  # Only send cookie over HTTPS
    SESSION_COOKIE_HTTPONLY=True,  # Prevent JavaScript access to session cookie
    SESSION_COOKIE_SAMESITE='Lax'  # Protect against CSRF
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
            df = pd.read_excel(file)
            logging.info(f"File caricato con successo: {file.filename}")
            
            if (df.empty):
                flash("Il file Excel è vuoto")
                return redirect(request.url)
            
            risultati = []
            all_series = {}
            
            # Process data in batches
            batch_size = 1000
            for start in range(0, len(df), batch_size):
                batch = df.iloc[start:start + batch_size]
                
                # Process each column in the batch
                for colonna in batch.columns:
                    try:
                        dati = pd.to_numeric(batch[colonna], errors='coerce').dropna().tolist()
                        if colonna in all_series:
                            all_series[colonna].extend(dati)
                        else:
                            all_series[colonna] = dati
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Errore di conversione nella colonna {colonna}: {str(e)}")
                    except Exception as e:
                        logging.error(f"Errore non previsto nella colonna {colonna}: {str(e)}")
            
            # Calcola la matrice di correlazione e t-test una sola volta
            matrice_correlazione_img = None
            matrice_ttest_img = None
            effect_size_img = None
            correlazioni = None
            t_tests = None
            legenda = {}
            if len(all_series) > 1:
                # Genera la matrice di correlazione
                matrice_correlazione_img, legenda = generate_correlation_matrix(all_series)
                correlazioni = StatisticheCalcolatore.calcola_correlazioni(all_series)
                
                # Calcola i t-test e genera i relativi grafici
                t_tests = StatisticheCalcolatore.calcola_ttest_coppie(all_series)
                
                # Calcola le dimensioni ottimali per la heatmap t-test
                n_vars = len(all_series)
                figsize_ttest = (min(12, max(8, n_vars * 1.2)), min(8, max(6, n_vars * 1.2)))
                
                # Genera heatmap t-test
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    ttest_legenda = StatisticheCalcolatore.crea_heatmap_ttest(
                        t_tests,
                        tmp.name,
                        use_etichette_brevi=True,
                        figsize=figsize_ttest
                    )
                    with open(tmp.name, 'rb') as f:
                        matrice_ttest_img = base64.b64encode(f.read()).decode('utf-8')
                    os.unlink(tmp.name)
                    
                    # Aggiorna la legenda con le etichette dei t-test
                    legenda.update(ttest_legenda)
                
                # Genera grafico effect size
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    effect_legenda = StatisticheCalcolatore.crea_effect_size_plot(
                        t_tests,
                        tmp.name,
                        use_etichette_brevi=True,
                        figsize=(12, 6)
                    )
                    with open(tmp.name, 'rb') as f:
                        effect_size_img = base64.b64encode(f.read()).decode('utf-8')
                    os.unlink(tmp.name)
                    
                    # Aggiorna la legenda con le etichette dell'effect size
                    legenda.update(effect_legenda)

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
                    
                    # Aggiungi la matrice di correlazione a tutte le serie
                    if matrice_correlazione_img:
                        stats_dict['plots']['correlation'] = matrice_correlazione_img
                        stats_dict['legenda'] = legenda
                    
                    # Aggiungi le correlazioni se disponibili
                    if correlazioni and colonna in correlazioni:
                        stats_dict['correlazioni'] = correlazioni[colonna]
                    
                    # Aggiungi i t-test se disponibili
                    if t_tests and colonna in t_tests:
                        stats_dict['t_tests'] = t_tests[colonna]
                        # Aggiungi l'interpretazione dell'effect size per ogni t-test
                        if isinstance(t_tests[colonna], dict):
                            for serie, test_result in t_tests[colonna].items():
                                if 'cohens_d' in test_result:
                                    test_result['effect_size'] = StatisticheCalcolatore.interpreta_cohens_d(test_result['cohens_d'])
                    
                    # Aggiungi i nuovi grafici alle statistiche
                    if matrice_ttest_img:
                        stats_dict['plots']['ttest'] = matrice_ttest_img
                    if effect_size_img:
                        stats_dict['plots']['effect_size'] = effect_size_img
                    
                    # Serializza i dati
                    stats_json = json.dumps(stats_dict)
                    valori_json = json.dumps(dati)
                    
                    # Crea il record nel database - Rimosso il campo risultato non necessario
                    calcolo = Calcolo(
                        nome=nome,
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
                    logging.error(f"Errore nell'elaborazione della serie {colonna}: {str(e)}")
                    flash(f"Errore nell'elaborazione della serie {colonna}: {str(e)}")
                    continue
            
            if not risultati:
                flash("Nessun dato numerico valido trovato nel file.")
                return redirect(request.url)
            
            try:
                db.session.commit()
                flash("Calcoli salvati con successo!", "success")
                return redirect(url_for('registro'))
            except Exception as e:
                db.session.rollback()
                logging.error(f"Errore nel salvataggio nel database: {str(e)}")
                flash("Errore nel salvataggio dei risultati nel database.")
                return redirect(request.url)
                
        except Exception as e:
            logging.error(f"Errore generale durante l'elaborazione del file: {str(e)}")
            flash(f"Errore durante l'elaborazione del file: {str(e)}")
            return redirect(request.url)
    
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
                        # Ensure all numeric fields are properly converted to float
                        if 'media' in stats:
                            stats['media'] = float(stats['media'])
                        if 'deviazione_standard_popolazione' in stats:
                            stats['deviazione_standard_popolazione'] = float(stats['deviazione_standard_popolazione'])
                        if 'count' in stats:
                            stats['count'] = int(stats['count'])
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
    
    return render_template('risultato.html', 
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