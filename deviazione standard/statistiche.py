import math
from collections import Counter
from typing import List, Dict, Union, Tuple, Optional

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
            Dict: Matrice di correlazione come dizionario di dizionari
        """
        import numpy as np
        import pandas as pd
        
        # Trova la lunghezza minima tra tutte le serie
        min_length = min(len(values) for values in serie_dati.values())
        
        # Tronca tutte le serie alla lunghezza minima
        adjusted_data = {
            name: values[:min_length] 
            for name, values in serie_dati.items()
        }
        
        # Converti il dizionario in DataFrame
        df = pd.DataFrame(adjusted_data)
        
        # Calcola la matrice di correlazione
        corr_matrix = df.corr().to_dict()
        
        return corr_matrix

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
        
        df_corr = pd.DataFrame(correlazioni)
        
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
    def crea_etichetta_breve(etichetta: str, max_len: int = 10) -> str:
        """
        Crea un'etichetta breve per i grafici.
        
        Args:
            etichetta: Etichetta originale
            max_len: Lunghezza massima dell'etichetta
            
        Returns:
            str: Etichetta abbreviata
        """
        if len(etichetta) <= max_len:
            return etichetta
            
        # Rimuovi spazi e caratteri speciali
        parole = etichetta.split()
        if len(parole) > 1:
            # Prendi le iniziali delle parole
            return ''.join(p[0].upper() for p in parole)
        else:
            # Tronca la parola
            return etichetta[:max_len-2] + '..'

    @staticmethod
    def esporta_pdf(nome_analisi: str, statistiche: dict, serie_dati: Dict[str, List[float]], 
                    percorso_output: str) -> str:
        """
        Esporta l'analisi completa in un file PDF.
        
        Args:
            nome_analisi: Nome dell'analisi
            statistiche: Dizionario con tutte le statistiche
            serie_dati: Dati originali
            percorso_output: Cartella dove salvare il PDF
            
        Returns:
            str: Percorso del file PDF creato
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        import os
        from datetime import datetime
        import tempfile
        import base64
        
        # Crea il nome del file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(percorso_output, f"analisi_{timestamp}.pdf")
        
        # Crea una directory temporanea per le immagini
        with tempfile.TemporaryDirectory() as temp_img_dir:
            # Inizializza il documento
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            # Titolo
            elements.append(Paragraph(f"Analisi Statistica: {nome_analisi}", styles['Title']))
            elements.append(Spacer(1, 12))
            
            # Data
            elements.append(Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
            elements.append(Spacer(1, 12))
            
            # Statistiche principali
            elements.append(Paragraph("Statistiche Principali", styles['Heading1']))
            data = []
            for k, v in statistiche.items():
                if k not in ['plots', 'legenda']:  # Escludiamo le immagini e la legenda
                    if isinstance(v, dict):
                        # Gestisci dizionari annidati
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, (int, float)):
                                data.append([f"{k} - {sub_k}", f"{sub_v:.4f}"])
                    elif isinstance(v, (int, float)):
                        data.append([k, f"{v:.4f}"])
                    else:
                        data.append([k, str(v)])
            
            if data:
                table = Table(data)
                table.setStyle([
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ])
                elements.append(table)
                elements.append(Spacer(1, 12))
            
            # Grafici
            if 'plots' in statistiche:
                elements.append(Paragraph("Visualizzazioni", styles['Heading1']))
                elements.append(Spacer(1, 12))
                
                img_paths = []  # Lista per tenere traccia dei file temporanei
                
                for plot_name, plot_data in statistiche['plots'].items():
                    # Salva l'immagine base64 come file temporaneo
                    temp_img_path = os.path.join(temp_img_dir, f"{plot_name}.png")
                    with open(temp_img_path, 'wb') as f:
                        f.write(base64.b64decode(plot_data))
                    img_paths.append(temp_img_path)
                    
                    # Aggiungi un titolo per il grafico
                    elements.append(Paragraph(plot_name.title(), styles['Heading2']))
                    elements.append(Spacer(1, 6))
                    
                    # Aggiungi l'immagine al PDF
                    img = Image(temp_img_path, width=6*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 12))
                
                # Se c'è una legenda per le correlazioni
                if 'legenda' in statistiche and statistiche['legenda']:
                    elements.append(Paragraph("Legenda Correlazioni", styles['Heading2']))
                    legend_data = [[k, v] for k, v in statistiche['legenda'].items()]
                    if legend_data:
                        legend_table = Table(legend_data)
                        legend_table.setStyle([
                            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black),
                            ('PADDING', (0, 0), (-1, -1), 6),
                        ])
                        elements.append(legend_table)
            
            # Genera il PDF
            doc.build(elements)
            
        return pdf_path