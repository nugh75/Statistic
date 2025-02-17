import math
from collections import Counter
from typing import List, Dict, Union, Tuple, Optional
import base64
from io import BytesIO
import PIL.Image
from reportlab.platypus import Image
from reportlab.lib.units import inch

class StatisticheCalcolatore:
    @staticmethod
    def valida_input(numeri: List[float]) -> None:
        """
        Valida l'input per i calcoli statistici.
        
        Args:
            numeri: Lista di numeri da validare
            
        Raises:
            ValueError: Se la lista è vuota o contiene valori non numerici
        """
        if not numeri:
            raise ValueError("La lista dei numeri non può essere vuota")
        if not all(isinstance(x, (int, float)) for x in numeri):
            raise ValueError("Tutti i valori devono essere numerici")

    @staticmethod
    def calcola_media(numeri: List[float]) -> float:
        """
        Calcola la media aritmetica di una lista di numeri.
        
        Args:
            numeri: Lista di numeri
            
        Returns:
            float: Media aritmetica
        """
        StatisticheCalcolatore.valida_input(numeri)
        return sum(numeri) / len(numeri)

    @staticmethod
    def calcola_mediana(numeri: List[float]) -> float:
        """
        Calcola la mediana di una lista di numeri.
        
        Args:
            numeri: Lista di numeri
            
        Returns:
            float: Mediana
        """
        StatisticheCalcolatore.valida_input(numeri)
        numeri_ordinati = sorted(numeri)
        n = len(numeri_ordinati)
        mid = n // 2
        
        if n % 2 == 0:
            return (numeri_ordinati[mid-1] + numeri_ordinati[mid]) / 2
        return numeri_ordinati[mid]

    @staticmethod
    def calcola_moda(numeri: List[float]) -> Union[float, List[float]]:
        """
        Calcola la moda di una lista di numeri.
        Se ci sono più valori con la stessa frequenza massima, ritorna una lista.
        
        Args:
            numeri: Lista di numeri
            
        Returns:
            Union[float, List[float]]: Moda o lista di mode
        """
        StatisticheCalcolatore.valida_input(numeri)
        conteggio = Counter(numeri)
        freq_max = max(conteggio.values())
        mode = [num for num, freq in conteggio.items() if freq == freq_max]
        return mode[0] if len(mode) == 1 else mode

    @staticmethod
    def calcola_varianza(numeri: List[float], popolazione: bool = True) -> float:
        """
        Calcola la varianza di una lista di numeri.
        
        Args:
            numeri: Lista di numeri
            popolazione: Se True calcola la varianza della popolazione, 
                      altrimenti del campione
            
        Returns:
            float: Varianza
        """
        StatisticheCalcolatore.valida_input(numeri)
        media = StatisticheCalcolatore.calcola_media(numeri)
        scarti_quadrati = [(x - media) ** 2 for x in numeri]
        divisore = len(numeri) if popolazione else len(numeri) - 1
        return sum(scarti_quadrati) / divisore

    @staticmethod
    def calcola_deviazione_standard(numeri: List[float], popolazione: bool = True) -> float:
        """
        Calcola la deviazione standard di una lista di numeri.
        
        Args:
            numeri: Lista di numeri
            popolazione: Se True calcola la deviazione standard della popolazione,
                      altrimenti del campione
            
        Returns:
            float: Deviazione standard
        """
        varianza = StatisticheCalcolatore.calcola_varianza(numeri, popolazione)
        return math.sqrt(varianza)

    @staticmethod
    def calcola_range(numeri: List[float]) -> float:
        """
        Calcola il range (differenza tra massimo e minimo) di una lista di numeri.
        
        Args:
            numeri: Lista di numeri
            
        Returns:
            float: Range
        """
        StatisticheCalcolatore.valida_input(numeri)
        return max(numeri) - min(numeri)

    @staticmethod
    def calcola_quartili(numeri: List[float]) -> Dict[str, float]:
        """
        Calcola i quartili di una lista di numeri.
        
        Args:
            numeri: Lista di numeri
            
        Returns:
            Dict[str, float]: Dizionario con Q1, Q2 (mediana) e Q3
        """
        StatisticheCalcolatore.valida_input(numeri)
        numeri_ordinati = sorted(numeri)
        n = len(numeri_ordinati)
        
        def trova_quartile(dati: List[float]) -> float:
            n_dati = len(dati)
            if n_dati % 2 == 0:
                return (dati[n_dati//2-1] + dati[n_dati//2]) / 2
            return dati[n_dati//2]
        
        # Q2 è la mediana
        q2 = StatisticheCalcolatore.calcola_mediana(numeri_ordinati)
        
        # Dividiamo la lista per Q1 e Q3
        meta = n // 2
        if n % 2 == 0:
            q1 = trova_quartile(numeri_ordinati[:meta])
            q3 = trova_quartile(numeri_ordinati[meta:])
        else:
            q1 = trova_quartile(numeri_ordinati[:meta])
            q3 = trova_quartile(numeri_ordinati[meta+1:])
            
        return {
            "Q1": q1,
            "Q2": q2,
            "Q3": q3
        }

    @staticmethod
    def calcola_min_max(numeri: List[float]) -> Dict[str, float]:
        """
        Trova il valore minimo e massimo di una lista di numeri.
        
        Args:
            numeri: Lista di numeri
            
        Returns:
            Dict[str, float]: Dizionario con minimo e massimo
        """
        StatisticheCalcolatore.valida_input(numeri)
        return {
            "min": min(numeri),
            "max": max(numeri)
        }

    @staticmethod
    def get_abbreviazioni() -> Dict[str, str]:
        """
        Restituisce un dizionario di abbreviazioni standard per le misure statistiche.
        
        Returns:
            Dict[str, str]: Dizionario con chiave nome completo e valore abbreviazione
        """
        return {
            "media": "x̄",  # media campionaria
            "mediana": "Me",
            "moda": "Mo",
            "deviazione_standard_popolazione": "σ",
            "deviazione_standard_campione": "s",
            "varianza_popolazione": "σ²",
            "varianza_campione": "s²",
            "range": "R",
            "quartili": {
                "Q1": "Q₁",
                "Q2": "Q₂",
                "Q3": "Q₃"
            },
            "min_max": {
                "min": "min",
                "max": "max"
            }
        }

    @staticmethod
    def calcola_tutte_statistiche(numeri: List[float], use_abbreviazioni: bool = False) -> Dict[str, Union[float, List[float], Dict[str, float]]]:
        """
        Calcola tutte le statistiche disponibili per una lista di numeri.
        
        Args:
            numeri: Lista di numeri
            use_abbreviazioni: Se True usa le abbreviazioni standard per le etichette
            
        Returns:
            Dict: Dizionario con tutte le statistiche calcolate
        """
        StatisticheCalcolatore.valida_input(numeri)
        risultati = {
            "count": len(numeri),
            "media": StatisticheCalcolatore.calcola_media(numeri),
            "mediana": StatisticheCalcolatore.calcola_mediana(numeri),
            "moda": StatisticheCalcolatore.calcola_moda(numeri),
            "deviazione_standard_popolazione": StatisticheCalcolatore.calcola_deviazione_standard(numeri, True),
            "deviazione_standard_campione": StatisticheCalcolatore.calcola_deviazione_standard(numeri, False),
            "varianza_popolazione": StatisticheCalcolatore.calcola_varianza(numeri, True),
            "varianza_campione": StatisticheCalcolatore.calcola_varianza(numeri, False),
            "range": StatisticheCalcolatore.calcola_range(numeri),
            "quartili": StatisticheCalcolatore.calcola_quartili(numeri),
            "min_max": StatisticheCalcolatore.calcola_min_max(numeri)
        }
        
        if use_abbreviazioni:
            abbreviazioni = StatisticheCalcolatore.get_abbreviazioni()
            risultati_abbreviati = {}
            for key, value in risultati.items():
                if key in abbreviazioni:
                    if isinstance(value, dict) and isinstance(abbreviazioni[key], dict):
                        # Gestisce nested dict come quartili e min_max
                        risultati_abbreviati[key] = {
                            abbreviazioni[key].get(k, k): v 
                            for k, v in value.items()
                        }
                    else:
                        risultati_abbreviati[abbreviazioni[key]] = value
                else:
                    risultati_abbreviati[key] = value
            return risultati_abbreviati
            
        return risultati

    @staticmethod
    def calcola_correlazioni(serie_dati: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Calcola le correlazioni tra tutte le serie di dati utilizzando numpy.
        Gestisce serie di lunghezze diverse troncando alla lunghezza minima comune.
        
        Args:
            serie_dati: Dizionario con nome serie come chiave e lista di valori come valore
            
        Returns:
            Dict: Matrice di correlazione strutturata come:
                 {serie1: {serie2: float}} dove float è il coefficiente di correlazione
        """
        import numpy as np
        import pandas as pd
        from scipy import stats
        
        # Trova la lunghezza minima tra tutte le serie
        min_length = min(len(values) for values in serie_dati.values())
        
        # Tronca tutte le serie alla lunghezza minima
        adjusted_data = {
            name: values[:min_length] 
            for name, values in serie_dati.items()
        }
        
        # Converti il dizionario in DataFrame
        df = pd.DataFrame(adjusted_data)
        
        # Dizionario per memorizzare correlazioni
        result = {}
        
        # Calcola correlazioni per ogni coppia di serie
        for col1 in df.columns:
            result[col1] = {}
            for col2 in df.columns:
                try:
                    if col1 != col2:
                        corr, _ = stats.pearsonr(df[col1], df[col2])
                        result[col1][col2] = float(corr)  # Store correlation coefficient
                    else:
                        result[col1][col2] = 1.0  # Perfect correlation with self
                except Exception as e:
                    print(f"Error calculating correlation between {col1} and {col2}: {str(e)}")
                    result[col1][col2] = 0.0  # Default to no correlation on error
        
        return result

    @staticmethod
    def calcola_cohens_d(serie1: List[float], serie2: List[float]) -> float:
        """
        Calcola Cohen's d come misura dell'effect size.
        
        Args:
            serie1: Prima serie di numeri
            serie2: Seconda serie di numeri
            
        Returns:
            float: Cohen's d effect size
        """
        # Calcola le medie
        mean1 = StatisticheCalcolatore.calcola_media(serie1)
        mean2 = StatisticheCalcolatore.calcola_media(serie2)
        
        # Calcola le varianze dei campioni
        var1 = StatisticheCalcolatore.calcola_varianza(serie1, popolazione=False)
        var2 = StatisticheCalcolatore.calcola_varianza(serie2, popolazione=False)
        
        # Calcola la deviazione standard combinata
        n1 = len(serie1)
        n2 = len(serie2)
        s_p = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Calcola Cohen's d
        d = (mean1 - mean2) / s_p if s_p != 0 else 0
        
        return float(d)

    @staticmethod
    def interpreta_cohens_d(d: float) -> str:
        """
        Interpreta il valore di Cohen's d.
        
        Args:
            d: Valore di Cohen's d
            
        Returns:
            str: Interpretazione dell'effect size
        """
        d = abs(d)  # Usa il valore assoluto per l'interpretazione
        if d < 0.2:
            return "trascurabile"
        elif d < 0.5:
            return "piccolo"
        elif d < 0.8:
            return "medio"
        else:
            return "grande"

    @staticmethod
    def calcola_ttest(serie1: List[float], serie2: List[float]) -> Dict[str, float]:
        """
        Calcola il t-test per due serie di dati.
        
        Args:
            serie1: Prima serie di numeri
            serie2: Seconda serie di numeri
            
        Returns:
            Dict[str, float]: Dizionario con statistica t, p-value e effect size (Cohen's d)
        """
        from scipy import stats
        
        # Calcola il t-test a due code per campioni indipendenti
        t_stat, p_value = stats.ttest_ind(serie1, serie2)
        
        # Calcola Cohen's d
        d = StatisticheCalcolatore.calcola_cohens_d(serie1, serie2)
        effect_size_interp = StatisticheCalcolatore.interpreta_cohens_d(d)
        
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(d),
            "effect_size": effect_size_interp
        }

    @staticmethod
    def calcola_ttest_coppie(serie_dati: Dict[str, List[float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calcola il t-test per ogni coppia di serie.
        
        Args:
            serie_dati: Dizionario con nome serie come chiave e lista di valori come valore
            
        Returns:
            Dict: Dizionario strutturato come:
                 {serie1: {serie2: {"t_statistic": float, "p_value": float, "cohens_d": float, "effect_size": str}}}
        """
        result = {}
        
        # Trova la lunghezza minima tra tutte le serie
        min_length = min(len(values) for values in serie_dati.values())
        
        # Tronca tutte le serie alla lunghezza minima
        adjusted_data = {
            name: values[:min_length] 
            for name, values in serie_dati.items()
        }
        
        # Calcola t-test per ogni coppia di serie
        for serie1 in adjusted_data:
            result[serie1] = {}
            for serie2 in adjusted_data:
                if serie1 != serie2:
                    try:
                        # Calcola t-test con effect size
                        test_results = StatisticheCalcolatore.calcola_ttest(
                            adjusted_data[serie1],
                            adjusted_data[serie2]
                        )
                        result[serie1][serie2] = test_results
                    except Exception as e:
                        print(f"Error calculating t-test between {serie1} and {serie2}: {str(e)}")
                        continue
        
        return result

    @staticmethod
    def crea_heatmap_correlazione(correlazioni: Dict[str, Dict[str, float]], percorso_file: str, use_etichette_brevi: bool = True, 
                                 figsize: Tuple[int, int] = (12, 8)) -> Dict[str, str]:
        """
        Crea una heatmap delle correlazioni e la salva come immagine.
        
        Args:
            correlazioni: Dizionario delle correlazioni
            percorso_file: Percorso dove salvare l'immagine
            use_etichette_brevi: Se True usa etichette alfabetiche
            figsize: Dimensioni della figura in pollici (larghezza, altezza)
            
        Returns:
            Dict[str, str]: Mappatura tra etichette brevi e originali
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import string
        
        # Estrai solo i coefficienti di correlazione dal dizionario
        df_data = {}
        for serie1, corr_dict in correlazioni.items():
            df_data[serie1] = {}
            for serie2, values in corr_dict.items():
                df_data[serie1][serie2] = values  # values è già il coefficiente
        
        df_corr = pd.DataFrame(df_data)
        
        # Crea mappatura etichette usando lettere dell'alfabeto
        legenda = {}
        if use_etichette_brevi:
            # Usa lettere maiuscole per le etichette
            lettere = list(string.ascii_uppercase)
            etichette_brevi = {col: lettere[i] for i, col in enumerate(df_corr.columns)}
            legenda = {v: k for k, v in etichette_brevi.items()}
            df_corr = df_corr.rename(columns=etichette_brevi)
            df_corr.index = df_corr.columns
        
        # Crea la heatmap
        plt.figure(figsize=figsize)
        
        # Imposta il formato dei numeri nella matrice
        heatmap = sns.heatmap(df_corr, 
                             annot=True,
                             cmap='coolwarm',
                             vmin=-1,
                             vmax=1,
                             center=0,
                             fmt='.2f',  # Mostra solo 2 decimali
                             annot_kws={'size': 10},  # Dimensione dei numeri
                             square=True)  # Celle quadrate
        
        # Ruota le etichette per maggiore leggibilità
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.title('Matrice di Correlazione')
        
        # Aggiusta il layout per evitare sovrapposizioni
        plt.tight_layout()
        
        # Salva la figura
        plt.savefig(percorso_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        return legenda

    @staticmethod
    def esporta_html_multiplo(titolo: str, all_statistics: list, series_data: dict, output_dir: str) -> str:
        """
        Genera un report HTML per multiple serie statistiche.
        
        Args:
            titolo: Titolo del report
            all_statistics: Lista di dizionari contenenti le statistiche per ogni serie
            series_data: Dizionario con i dati grezzi delle serie
            output_dir: Directory dove salvare il report e le immagini
            
        Returns:
            str: Percorso del file HTML generato
        """
        import os
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        from datetime import datetime
        
        # Crea la directory per le immagini se non esiste
        img_dir = os.path.join(output_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        
        # Salva le immagini dei plot come file PNG
        for i, stats in enumerate(all_statistics):
            if 'plots' in stats:
                plots_dir = {}
                for plot_type, plot_data in stats['plots'].items():
                    try:
                        # Verifica se il dato è una stringa base64
                        if isinstance(plot_data, str):
                            # Se inizia con data:image/png;base64,
                            if plot_data.startswith('data:image/png;base64,'):
                                img_data = base64.b64decode(plot_data.split(',')[1])
                            # Se è già una stringa base64 senza il prefisso
                            else:
                                img_data = base64.b64decode(plot_data)
                            
                            # Crea un nome file univoco per l'immagine
                            img_filename = f'serie_{i+1}_{plot_type}.png'
                            img_path = os.path.join(img_dir, img_filename)
                            
                            # Salva l'immagine
                            with open(img_path, 'wb') as f:
                                f.write(img_data)
                            
                            # Aggiorna il riferimento nel dizionario delle statistiche
                            plots_dir[plot_type] = f'images/{img_filename}'
                        else:
                            # Se il plot_data non è una stringa, lo ignoriamo
                            print(f"Warning: plot data for {plot_type} is not a string")
                            continue
                            
                    except Exception as e:
                        print(f"Error processing image {plot_type} for series {i+1}: {str(e)}")
                        continue
                
                stats['plots'] = plots_dir
        
        # Crea il template HTML
        html_template = '''
        <!DOCTYPE html>
        <html lang="it">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ titolo }}</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f7fa;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .serie-container {
                    margin-bottom: 40px;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .stats-section {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 6px;
                }
                .stats-section h4 {
                    color: #3498db;
                    margin-top: 0;
                    margin-bottom: 10px;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 5px;
                }
                .stats-section ul {
                    list-style-type: none;
                    padding: 0;
                    margin: 0;
                }
                .stats-section li {
                    margin-bottom: 8px;
                    padding: 8px;
                    background-color: white;
                    border-radius: 4px;
                }
                .plots-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .plot-card {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }
                .plot-card img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .note-box {
                    background-color: #fff3cd;
                    color: #856404;
                    padding: 15px;
                    margin-top: 15px;
                    border-radius: 4px;
                    border-left: 4px solid #ffeeba;
                }
                .correlations-section {
                    margin-top: 20px;
                }
                .correlations-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 10px;
                }
                .correlation-item {
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 4px;
                    display: flex;
                    justify-content: space-between;
                }
                .footer {
                    margin-top: 40px;
                    text-align: center;
                    color: #666;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{{ titolo }}</h1>
                <p>Report generato il {{ data_generazione }}</p>
                
                {% for stats in all_statistics %}
                <div class="serie-container">
                    <h2>{{ stats.nome_calcolo }}</h2>
                    <h3>Serie: {{ stats.serie_nome }}</h3>
                    
                    {% if stats.note %}
                    <div class="note-box">
                        <strong>Note:</strong> {{ stats.note }}
                    </div>
                    {% endif %}
                    
                    <div class="stats-grid">
                        <div class="stats-section">
                            <h4>Informazioni Dataset</h4>
                            <ul>
                                <li>Numero di valori: <strong>{{ stats.count }}</strong></li>
                            </ul>
                        </div>
                        
                        <div class="stats-section">
                            <h4>Valori Principali</h4>
                            <ul>
                                <li>Minimo: <strong>{{ "%.4f"|format(stats.min_max.min) }}</strong></li>
                                <li>Q1 (25° percentile): <strong>{{ "%.4f"|format(stats.quartili.Q1) }}</strong></li>
                                <li>Mediana (Q2): <strong>{{ "%.4f"|format(stats.mediana) }}</strong></li>
                                <li>Media: <strong>{{ "%.4f"|format(stats.media) }}</strong></li>
                                <li>Q3 (75° percentile): <strong>{{ "%.4f"|format(stats.quartili.Q3) }}</strong></li>
                                <li>Massimo: <strong>{{ "%.4f"|format(stats.min_max.max) }}</strong></li>
                            </ul>
                        </div>
                        
                        <div class="stats-section">
                            <h4>Misure di Dispersione</h4>
                            <ul>
                                <li>Dev. Std. (pop.): <strong>{{ "%.6f"|format(stats.deviazione_standard_popolazione) }}</strong></li>
                                <li>Dev. Std. (camp.): <strong>{{ "%.6f"|format(stats.deviazione_standard_campione) }}</strong></li>
                                <li>Range: <strong>{{ "%.4f"|format(stats.range) }}</strong></li>
                            </ul>
                        </div>
                        
                        <div class="stats-section">
                            <h4>Altri Indicatori</h4>
                            <ul>
                                <li>Moda: 
                                    <strong>
                                    {% if stats.moda is string or stats.moda is number %}
                                        {{ "%.4f"|format(stats.moda) }}
                                    {% else %}
                                        {{ stats.moda|map('format_float')|join(", ") }}
                                    {% endif %}
                                    </strong>
                                </li>
                                <li>Varianza (pop.): <strong>{{ "%.6f"|format(stats.varianza_popolazione) }}</strong></li>
                                <li>Varianza (camp.): <strong>{{ "%.6f"|format(stats.varianza_campione) }}</strong></li>
                            </ul>
                        </div>
                    </div>
                    
                    {% if stats.plots %}
                    <div class="visualizations-section">
                        <h3>Visualizzazioni Statistiche</h3>
                        <div class="plots-grid">
                            {% if stats.plots.histogram %}
                            <div class="plot-card">
                                <h4>Istogramma con KDE e Distribuzione Normale</h4>
                                <img src="{{ stats.plots.histogram }}" alt="Histogram">
                            </div>
                            {% endif %}
                            
                            {% if stats.plots.boxplot %}
                            <div class="plot-card">
                                <h4>Box Plot</h4>
                                <img src="{{ stats.plots.boxplot }}" alt="Box Plot">
                            </div>
                            {% endif %}
                            
                            {% if stats.plots.qqplot %}
                            <div class="plot-card">
                                <h4>Q-Q Plot (Test di Normalità)</h4>
                                <img src="{{ stats.plots.qqplot }}" alt="Q-Q Plot">
                            </div>
                            {% endif %}
                            
                            {% if stats.plots.correlation %}
                            <div class="plot-card">
                                <h4>Matrice di Correlazione</h4>
                                <img src="{{ stats.plots.correlation }}" alt="Correlation Matrix">
                            </div>
                            {% endif %}
                        </div>
                    </div>
                    {% endif %}
                    
                    {% if stats.correlazioni %}
                    <div class="correlations-section">
                        <h3>Correlazioni con altre serie</h3>
                        <div class="correlations-grid">
                            {% for altra_serie, correlazione in stats.correlazioni.items() %}
                                {% if altra_serie != stats.serie_nome %}
                                <div class="correlation-item">
                                    <span>{{ altra_serie }}:</span>
                                    <strong>{{ "%.4f"|format(correlazione) }}</strong>
                                </div>
                                {% endif %}
                            {% endfor %}
                        </div>
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
                
                <div class="footer">
                    <p>Report generato automaticamente da StatisticheCalcolatore</p>
                </div>
            </div>
            
            <script>
                // Aggiungi funzione per formattare i numeri float
                function formatFloat(value) {
                    return value.toFixed(4);
                }
            </script>
        </body>
        </html>
        '''
        
        # Crea l'ambiente Jinja2
        env = Environment(autoescape=select_autoescape(['html', 'xml']))
        
        # Aggiungi il filtro format_float
        env.filters['format_float'] = lambda x: f"{float(x):.4f}"
        
        # Compila il template
        template = env.from_string(html_template)
        
        # Genera l'HTML
        html_content = template.render(
            titolo=titolo,
            all_statistics=all_statistics,
            data_generazione=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
        
        # Salva il file HTML
        html_path = os.path.join(output_dir, 'report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path