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
    def calcola_correlazioni(serie_dati: Dict[str, List[float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calcola le correlazioni tra tutte le serie di dati utilizzando numpy.
        Gestisce serie di lunghezze diverse troncando alla lunghezza minima comune.
        Calcola anche la significatività statistica (p-value) per ogni correlazione.
        
        Args:
            serie_dati: Dizionario con nome serie come chiave e lista di valori come valore
            
        Returns:
            Dict: Matrice di correlazione con coefficienti e p-values strutturata come:
                 {serie1: {serie2: {'coefficiente': float, 'p_value': float}}}
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
        
        # Dizionario per memorizzare correlazioni e p-values
        result = {}
        
        # Calcola correlazioni e p-values per ogni coppia di serie
        for col1 in df.columns:
            result[col1] = {}
            for col2 in df.columns:
                try:
                    if col1 != col2:
                        corr, p_value = stats.pearsonr(df[col1], df[col2])
                        result[col1][col2] = {
                            'coefficiente': float(corr),  # Ensure float type
                            'p_value': float(p_value)     # Ensure float type
                        }
                    else:
                        result[col1][col2] = {
                            'coefficiente': 1.0,
                            'p_value': 0.0
                        }
                except Exception as e:
                    print(f"Error calculating correlation between {col1} and {col2}: {str(e)}")
                    result[col1][col2] = {
                        'coefficiente': 0.0,
                        'p_value': 1.0
                    }
        
        return result

    @staticmethod
    def crea_heatmap_correlazione(correlazioni: Dict[str, Dict[str, Dict[str, float]]], percorso_file: str, use_etichette_brevi: bool = True, 
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
                df_data[serie1][serie2] = values['coefficiente']
        
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
    def base64_to_image(base64_str: str) -> Image:
        """Convert base64 string to ReportLab Image object."""
        try:
            # Decode base64 to bytes
            img_data = base64.b64decode(base64_str)
            
            # Create a BytesIO object
            img_buffer = BytesIO(img_data)
            
            # Open with PIL to verify image and get size
            pil_img = PIL.Image.open(img_buffer)
            width, height = pil_img.size
            
            # Create a new BytesIO for the final image
            final_buffer = BytesIO()
            pil_img.save(final_buffer, format='PNG')
            final_buffer.seek(0)
            
            # Create ReportLab Image
            img = Image(final_buffer)
            
            # Scale image while maintaining aspect ratio
            aspect = height / width
            img.drawWidth = 6 * inch
            img.drawHeight = 6 * inch * aspect
            
            return img
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

    @staticmethod
    def esporta_pdf(nome_analisi: str, statistiche: dict, serie_dati: Dict[str, List[float]], 
                    percorso_output: str) -> str:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        import os
        from datetime import datetime
        import base64
        from io import BytesIO
        import PIL.Image

        # Material Design Blue Colors
        primary_color = HexColor('#2196F3')
        secondary_color = HexColor('#1976D2')
        text_color = HexColor('#212121')

        # Create custom styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='MaterialTitle',
            parent=styles['Title'],
            textColor=secondary_color,
            spaceAfter=30,
            fontSize=24,
            leading=30
        ))
        styles.add(ParagraphStyle(
            name='MaterialHeading1',
            parent=styles['Heading1'],
            textColor=primary_color,
            fontSize=18,
            spaceAfter=16
        ))
        styles.add(ParagraphStyle(
            name='MaterialHeading2',
            parent=styles['Heading2'],
            textColor=primary_color,
            fontSize=16,
            spaceAfter=12
        ))
        styles.add(ParagraphStyle(
            name='MaterialNormal',
            parent=styles['Normal'],
            textColor=text_color,
            fontSize=11,
            spaceAfter=8
        ))

        # Create PDF with proper margins
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(percorso_output, f"analisi_{timestamp}.pdf")

        # Use wider page margins for better readability
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            leftMargin=inch*0.75,
            rightMargin=inch*0.75,
            topMargin=inch*0.75,
            bottomMargin=inch*0.75
        )
        elements = []

        # Header
        elements.append(Paragraph(f"Analisi Statistica: {nome_analisi}", styles['MaterialTitle']))
        elements.append(Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['MaterialNormal']))
        elements.append(Spacer(1, 20))

        # Statistics Table with improved layout
        elements.append(Paragraph("Statistiche Principali", styles['MaterialHeading1']))
        elements.append(Spacer(1, 10))

        # Prepare table data with wrapped text
        data = []
        headers = [["Misura", "Valore"]]
        table_data = []

        for k, v in statistiche.items():
            if k not in ['plots', 'legenda']:
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            # Break long measurement names with word wrapping
                            measure = Paragraph(f"{k} - {sub_k}", styles['MaterialNormal'])
                            value = Paragraph(f"{sub_v:.4f}", styles['MaterialNormal'])
                            table_data.append([measure, value])
                elif isinstance(v, (int, float)):
                    measure = Paragraph(k, styles['MaterialNormal'])
                    value = Paragraph(f"{v:.4f}", styles['MaterialNormal'])
                    table_data.append([measure, value])
                else:
                    measure = Paragraph(k, styles['MaterialNormal'])
                    value = Paragraph(str(v), styles['MaterialNormal'])
                    table_data.append([measure, value])

        data = headers + table_data
        if data:
            # Increased column widths and added word wrapping
            table = Table(data, colWidths=[4.5*inch, 2.5*inch], rowHeights=None)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, 0), 12),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F5F5F5')]),
                ('TEXTCOLOR', (0, 1), (-1, -1), text_color),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('PADDING', (0, 0), (-1, -1), 12),  # Increased padding
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Vertical centering
                ('WORDWRAP', (0, 0), (-1, -1), True),    # Enable word wrapping
            ]))
            elements.append(table)
            elements.append(Spacer(1, 20))

        # Plots
        if 'plots' in statistiche:
            elements.append(Paragraph("Visualizzazioni Statistiche", styles['MaterialHeading1']))
            elements.append(Spacer(1, 15))

            plot_titles = {
                'histogram': 'Istogramma con KDE e Distribuzione Normale',
                'boxplot': 'Box Plot',
                'qqplot': 'Q-Q Plot (Test di Normalità)'
            }

            # Add individual plots
            for plot_name, plot_data in statistiche['plots'].items():
                if plot_name != 'correlation':
                    elements.append(Paragraph(plot_titles.get(plot_name, plot_name.title()), 
                                           styles['MaterialHeading2']))
                    elements.append(Spacer(1, 8))
                    
                    img = StatisticheCalcolatore.base64_to_image(plot_data)
                    if img:
                        elements.append(img)
                        elements.append(Spacer(1, 20))

            # Add correlation matrix if available
            if 'correlation' in statistiche['plots']:
                elements.append(Paragraph("Matrice di Correlazione", styles['MaterialHeading2']))
                elements.append(Spacer(1, 8))
                
                img = StatisticheCalcolatore.base64_to_image(statistiche['plots']['correlation'])
                if img:
                    elements.append(img)
                    elements.append(Spacer(1, 15))

                # Add correlation legend if available
                if 'legenda' in statistiche:
                    elements.append(Paragraph("Legenda delle Serie", styles['MaterialHeading2']))
                    elements.append(Spacer(1, 8))
                    
                    legend_data = [["Etichetta", "Nome Serie"]] + \
                                [[k, v] for k, v in statistiche['legenda'].items()]
                    
                    legend_table = Table(legend_data, colWidths=[1.5*inch, 4.5*inch])
                    legend_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F5F5F5')]),
                        ('TEXTCOLOR', (0, 1), (-1, -1), text_color),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('PADDING', (0, 0), (-1, -1), 8),
                    ]))
                    elements.append(legend_table)

            # Add correlation strength guide
            if 'correlazioni' in statistiche:
                elements.append(Paragraph("Correlazioni Statisticamente Significative (p < 0.05)", styles['MaterialHeading2']))
                elements.append(Spacer(1, 8))

                # Add strength guide
                guide_data = [["Forza della Correlazione", "Range"]]
                guide_rows = [
                    ["Molto forte positiva", "≥ 0.9"],
                    ["Forte positiva", "0.7 - 0.9"],
                    ["Moderata positiva", "0.5 - 0.7"],
                    ["Moderata negativa", "-0.7 - -0.5"],
                    ["Forte negativa", "-0.9 - -0.7"],
                    ["Molto forte negativa", "≤ -0.9"]
                ]
                guide_table = Table([["Guida Interpretazione Correlazioni"]], colWidths=[7*inch])
                guide_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('PADDING', (0, 0), (-1, 0), 8),
                ]))
                elements.append(guide_table)
                elements.append(Spacer(1, 8))

                strength_table = Table(guide_data + guide_rows, colWidths=[3.5*inch, 3.5*inch])
                strength_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f8f9fa')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('PADDING', (0, 0), (-1, -1), 8),
                ]))
                elements.append(strength_table)
                elements.append(Spacer(1, 15))

                # Add significant correlations
                significant_data = [["Serie 1", "Serie 2", "Coefficiente", "P-value", "Forza"]]
                significant_rows = []

                for serie1, correlazioni in statistiche['correlazioni'].items():
                    if isinstance(correlazioni, dict):
                        for serie2, values in correlazioni.items():
                            if (serie1 != serie2 and 
                                isinstance(values, dict) and 
                                'coefficiente' in values and 
                                'p_value' in values and 
                                values['p_value'] < 0.05):
                                
                                coef = values['coefficiente']
                                strength = ""
                                bg_color = colors.white

                                if abs(coef) >= 0.9:
                                    strength = "Molto forte " + ("positiva" if coef > 0 else "negativa")
                                    bg_color = HexColor('#c6f6d5' if coef > 0 else '#fed7d7')
                                elif abs(coef) >= 0.7:
                                    strength = "Forte " + ("positiva" if coef > 0 else "negativa")
                                    bg_color = HexColor('#d4f5d4' if coef > 0 else '#fdd')
                                elif abs(coef) >= 0.5:
                                    strength = "Moderata " + ("positiva" if coef > 0 else "negativa")
                                    bg_color = HexColor('#e6ffe6' if coef > 0 else '#fff2f2')

                                if strength:  # Solo correlazioni almeno moderate
                                    significant_rows.append([
                                        serie1,
                                        serie2,
                                        f"{values['coefficiente']:.3f}",
                                        f"{values['p_value']:.4f}",
                                        strength,
                                        bg_color
                                    ])

                if significant_rows:
                    # Sort by absolute correlation coefficient
                    significant_rows.sort(key=lambda x: abs(float(x[2])), reverse=True)
                    
                    # Remove background color from data before creating table
                    table_data = significant_data + [[row[0], row[1], row[2], row[3], row[4]] for row in significant_rows]
                    
                    table = Table(table_data, colWidths=[1.4*inch, 1.4*inch, 1.2*inch, 1.2*inch, 1.8*inch])
                    
                    # Create style with background colors
                    style = [
                        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('PADDING', (0, 0), (-1, -1), 8),
                    ]
                    
                    # Add background colors for each row
                    for i, row in enumerate(significant_rows, 1):
                        style.append(('BACKGROUND', (0, i), (-1, i), row[5]))
                    
                    table.setStyle(TableStyle(style))
                    elements.append(table)
                else:
                    elements.append(Paragraph("Nessuna correlazione statisticamente significativa trovata.", 
                                           styles['MaterialNormal']))

                elements.append(Spacer(1, 20))

        try:
            # Build the PDF
            doc.build(elements)
            return pdf_path
        except Exception as e:
            print(f"Errore durante la generazione del PDF: {str(e)}")
            raise

    @staticmethod
    def esporta_pdf_multiplo(titolo: str, statistiche_multiple: List[dict], serie_dati: Dict[str, List[float]], 
                            percorso_output: str) -> str:
        """
        Crea un PDF contenente multiple analisi statistiche nell'ordine specificato.
        
        Args:
            titolo: Titolo generale del documento
            statistiche_multiple: Lista di dizionari contenenti le statistiche per ogni serie
            serie_dati: Dizionario con nome serie come chiave e lista di valori come valore
            percorso_output: Percorso dove salvare il PDF
            
        Returns:
            str: Percorso del file PDF generato
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        import os
        from datetime import datetime

        # Material Design Colors
        primary_color = HexColor('#2196F3')
        secondary_color = HexColor('#1976D2')
        text_color = HexColor('#212121')

        # Create custom styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='MaterialTitle',
            parent=styles['Title'],
            textColor=secondary_color,
            spaceAfter=30,
            fontSize=24,
            leading=30
        ))
        styles.add(ParagraphStyle(
            name='MaterialHeading1',
            parent=styles['Heading1'],
            textColor=primary_color,
            fontSize=18,
            spaceAfter=16
        ))
        styles.add(ParagraphStyle(
            name='MaterialHeading2',
            parent=styles['Heading2'],
            textColor=primary_color,
            fontSize=16,
            spaceAfter=12
        ))
        styles.add(ParagraphStyle(
            name='MaterialNormal',
            parent=styles['Normal'],
            textColor=text_color,
            fontSize=11,
            spaceAfter=8
        ))

        # Create PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(percorso_output, f"analisi_multiple_{timestamp}.pdf")
        
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            leftMargin=inch*0.75,
            rightMargin=inch*0.75,
            topMargin=inch*0.75,
            bottomMargin=inch*0.75
        )
        elements = []

        # Main Title
        elements.append(Paragraph(titolo, styles['MaterialTitle']))
        elements.append(Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['MaterialNormal']))
        elements.append(Spacer(1, 20))
        
        # Table of Contents
        elements.append(Paragraph("Indice delle Serie", styles['MaterialHeading1']))
        elements.append(Spacer(1, 10))
        
        toc_data = [["N°", "Serie", "Nome Analisi"]]
        for i, stats in enumerate(statistiche_multiple, 1):
            toc_data.append([
                str(i),
                stats['serie_nome'],
                stats['nome_calcolo']
            ])
        
        toc_table = Table(toc_data, colWidths=[0.5*inch, 3*inch, 3.5*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F5F5F5')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(toc_table)
        elements.append(PageBreak())

        # Add each series analysis
        for i, statistiche in enumerate(statistiche_multiple, 1):
            # Series Title
            elements.append(Paragraph(f"Serie {i}: {statistiche['nome_calcolo']}", styles['MaterialHeading1']))
            elements.append(Paragraph(f"Nome Serie: {statistiche['serie_nome']}", styles['MaterialNormal']))
            if statistiche.get('note'):
                elements.append(Paragraph(f"Note: {statistiche['note']}", styles['MaterialNormal']))
            elements.append(Spacer(1, 20))

            # Statistics Table
            elements.append(Paragraph("Statistiche", styles['MaterialHeading2']))
            elements.append(Spacer(1, 10))

            table_data = []
            headers = [["Misura", "Valore"]]
            stats_rows = []

            for k, v in statistiche.items():
                if k not in ['plots', 'legenda', 'nome_calcolo', 'serie_nome', 'note', 'correlazioni']:
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, (int, float)):
                                measure = Paragraph(f"{k} - {sub_k}", styles['MaterialNormal'])
                                value = Paragraph(f"{sub_v:.4f}", styles['MaterialNormal'])
                                stats_rows.append([measure, value])
                    elif isinstance(v, (int, float)):
                        measure = Paragraph(k, styles['MaterialNormal'])
                        value = Paragraph(f"{v:.4f}", styles['MaterialNormal'])
                        stats_rows.append([measure, value])
                    else:
                        measure = Paragraph(k, styles['MaterialNormal'])
                        value = Paragraph(str(v), styles['MaterialNormal'])
                        stats_rows.append([measure, value])

            table_data = headers + stats_rows
            if table_data:
                table = Table(table_data, colWidths=[4*inch, 3*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F5F5F5')]),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 20))

            # Plots
            if 'plots' in statistiche:
                elements.append(Paragraph("Visualizzazioni", styles['MaterialHeading2']))
                elements.append(Spacer(1, 10))

                plot_titles = {
                    'histogram': 'Istogramma con KDE e Distribuzione Normale',
                    'boxplot': 'Box Plot',
                    'qqplot': 'Q-Q Plot (Test di Normalità)'
                }

                for plot_name, plot_data in statistiche['plots'].items():
                    if plot_name != 'correlation':
                        elements.append(Paragraph(plot_titles.get(plot_name, plot_name.title()), 
                                               styles['MaterialHeading2']))
                        elements.append(Spacer(1, 8))
                        
                        img = StatisticheCalcolatore.base64_to_image(plot_data)
                        if img:
                            elements.append(img)
                            elements.append(Spacer(1, 20))

            # Add page break between series
            if i < len(statistiche_multiple):
                elements.append(PageBreak())

        # Add correlation matrix at the end if available
        if len(statistiche_multiple) > 1 and 'plots' in statistiche_multiple[0] and 'correlation' in statistiche_multiple[0]['plots']:
            elements.append(PageBreak())
            elements.append(Paragraph("Matrice di Correlazione tra Serie", styles['MaterialHeading1']))
            elements.append(Spacer(1, 15))
            
            img = StatisticheCalcolatore.base64_to_image(statistiche_multiple[0]['plots']['correlation'])
            if img:
                elements.append(img)
                elements.append(Spacer(1, 15))

            # Add legend if available
            if 'legenda' in statistiche_multiple[0]:
                elements.append(Paragraph("Legenda delle Serie", styles['MaterialHeading2']))
                elements.append(Spacer(1, 8))
                
                legend_data = [["Etichetta", "Nome Serie"]] + \
                            [[k, v] for k, v in statistiche_multiple[0]['legenda'].items()]
                
                legend_table = Table(legend_data, colWidths=[1.5*inch, 5.5*inch])
                legend_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#F5F5F5')]),
                ]))
                elements.append(legend_table)

        try:
            doc.build(elements)
            return pdf_path
        except Exception as e:
            print(f"Errore durante la generazione del PDF multiplo: {str(e)}")
            raise

    @staticmethod
    def esporta_html_multiplo(titolo: str, statistiche_multiple: List[dict], serie_dati: Dict[str, List[float]], 
                            percorso_output: str) -> str:
        """
        Crea un file HTML contenente multiple analisi statistiche nell'ordine specificato.
        I grafici vengono salvati come file separati nella cartella images/.
        
        Args:
            titolo: Titolo generale del documento
            statistiche_multiple: Lista di dizionari contenenti le statistiche per ogni serie
            serie_dati: Dizionario con nome serie come chiave e lista di valori come valore
            percorso_output: Percorso dove salvare i file HTML e immagini
            
        Returns:
            str: Percorso del file HTML generato
        """
        import os
        from datetime import datetime
        import base64
        from pathlib import Path

        # Crea directory per il report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(percorso_output, f"report_{timestamp}")
        images_dir = os.path.join(report_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Copia il CSS in una directory locale
        css_content = """
        /* Reset e stili di base */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Roboto', sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background-color: #f5f7fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        /* Header e titoli */
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 2.5rem;
            font-size: 2.5rem;
            font-weight: 500;
            position: relative;
            padding-bottom: 1rem;
        }
        
        h1:after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 4px;
            background: linear-gradient(to right, #3498db, #2980b9);
            border-radius: 2px;
        }
        
        h2 {
            color: #34495e;
            margin-bottom: 1rem;
        }
        
        h3 {
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }
        
        h4 {
            color: #3498db;
            margin-bottom: 0.5rem;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.3rem;
        }
        
        .data {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        
        .toc {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .toc h2 {
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        .toc-content {
            padding-left: 1rem;
        }
        
        .toc-content a {
            color: #3498db;
            text-decoration: none;
            line-height: 1.8;
        }
        
        .toc-content a:hover {
            color: #2980b9;
            text-decoration: underline;
        }
        
        .serie-analysis {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .note {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            margin: 1rem 0;
            border-left: 3px solid #3498db;
        }
        
        .stats-grid-compact {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }
        
        .stats-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }
        
        .stats-section h4 {
            color: #3498db;
            margin-bottom: 1rem;
        }
        
        .stats-section ul {
            list-style: none;
            margin: 0;
            padding: 0;
        }
        
        .stats-section li {
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            background: white;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }
        
        .stats-section strong {
            float: right;
            color: #2c3e50;
        }
        
        .visualizations-section {
            margin-top: 2rem;
        }
        
        .plots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }
        
        .plot-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .plot-card h5 {
            color: #2c3e50;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .plot-card img {
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .correlation-section {
            margin-top: 3rem;
            padding: 2rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .correlation-legend {
            margin-top: 2rem;
        }
        
        .legend-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        .legend-table th,
        .legend-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }
        
        .legend-table th {
            background: #f8f9fa;
            font-weight: 500;
        }
        
        @media print {
            body {
                background: white;
            }
            
            .container {
                max-width: none;
                padding: 0;
            }
            
            .serie-analysis,
            .correlation-section {
                break-inside: avoid;
                page-break-inside: avoid;
            }
            
            .plot-card img {
                max-width: 100%;
                height: auto;
            }
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .stats-grid-compact {
                grid-template-columns: 1fr;
            }
            
            .plots-grid {
                grid-template-columns: 1fr;
            }
        }
        """
        
        with open(os.path.join(report_dir, "style.css"), "w", encoding='utf-8') as f:
            f.write(css_content)

        # Template HTML
        html_template = """
        <!DOCTYPE html>
        <html lang="it">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
            <link rel="stylesheet" href="style.css">
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <p class="data">Data: {date}</p>
                
                <div class="toc">
                    <h2>Indice delle Serie</h2>
                    <div class="toc-content">
                        {toc}
                    </div>
                </div>

                <div class="serie-analyses">
                    {analyses}
                </div>
            </div>
        </body>
        </html>
        """

        # Genera indice
        toc_items = []
        analyses_html = []
        
        for i, stats in enumerate(statistiche_multiple, 1):
            serie_id = f"serie_{i}"
            toc_items.append(f'<p><a href="#{serie_id}">Serie {i}: {stats["serie_nome"]} - {stats["nome_calcolo"]}</a></p>')
            
            # Salva i grafici come file separati
            plots_html = []
            if 'plots' in stats:
                for plot_name, plot_data in stats['plots'].items():
                    if plot_name != 'correlation':
                        img_filename = f"{serie_id}_{plot_name}.png"
                        img_path = os.path.join(images_dir, img_filename)
                        
                        # Decodifica e salva l'immagine
                        img_data = base64.b64decode(plot_data)
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                        
                        plot_title = {
                            'histogram': 'Istogramma con KDE e Distribuzione Normale',
                            'boxplot': 'Box Plot',
                            'qqplot': 'Q-Q Plot (Test di Normalità)'
                        }.get(plot_name, plot_name.title())
                        
                        plots_html.append(f"""
                        <div class="plot-card">
                            <h5>{plot_title}</h5>
                            <img src="images/{img_filename}" alt="{plot_title}">
                        </div>
                        """)

            # Genera HTML per la singola serie
            serie_html = f"""
            <div id="{serie_id}" class="serie-analysis">
                <h2>Serie {i}: {stats['nome_calcolo']}</h2>
                <p class="serie-nome">Nome Serie: {stats['serie_nome']}</p>
                
                {'<p class="note">Note: ' + stats['note'] + '</p>' if stats.get('note') else ''}
                
                <div class="stats-container">
                    <h3>Statistiche</h3>
                    <div class="stats-grid-compact">
                        <!-- Statistiche principali -->
                        {StatisticheCalcolatore._genera_html_statistiche(stats)}
                    </div>
                </div>

                <div class="visualizations-section">
                    <h3>Visualizzazioni</h3>
                    <div class="plots-grid">
                        {''.join(plots_html)}
                    </div>
                </div>
            </div>
            """
            analyses_html.append(serie_html)

        # Gestisci matrice di correlazione se presente
        if (len(statistiche_multiple) > 1 and 'plots' in statistiche_multiple[0] and 
            'correlation' in statistiche_multiple[0]['plots']):
            
            # Salva matrice di correlazione
            correlation_filename = "correlation_matrix.png"
            correlation_path = os.path.join(images_dir, correlation_filename)
            correlation_data = base64.b64decode(statistiche_multiple[0]['plots']['correlation'])
            with open(correlation_path, 'wb') as f:
                f.write(correlation_data)
            
            # Aggiungi sezione correlazioni
            correlation_html = f"""
            <div class="correlation-section">
                <h2>Matrice di Correlazione tra Serie</h2>
                <div class="correlation-plot">
                    <img src="images/{correlation_filename}" alt="Correlation Matrix">
                </div>
            """
            
            # Aggiungi legenda se presente
            if 'legenda' in statistiche_multiple[0]:
                correlation_html += """
                <div class="correlation-legend">
                    <h3>Legenda delle Serie</h3>
                    <table class="legend-table">
                        <thead>
                            <tr>
                                <th>Etichetta</th>
                                <th>Nome Serie</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                for lettera, nome in statistiche_multiple[0]['legenda'].items():
                    correlation_html += f"""
                            <tr>
                                <td><strong>{lettera}</strong></td>
                                <td>{nome}</td>
                            </tr>
                    """
                correlation_html += """
                        </tbody>
                    </table>
                </div>
                """
            
            correlation_html += "</div>"
            analyses_html.append(correlation_html)

        # Assembla il documento HTML finale
        html_content = html_template.format(
            title=titolo,
            date=datetime.now().strftime('%d/%m/%Y %H:%M'),
            toc='\n'.join(toc_items),
            analyses='\n'.join(analyses_html)
        )

        # Salva il file HTML
        html_path = os.path.join(report_dir, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path

    @staticmethod
    def _genera_html_statistiche(stats: dict) -> str:
        """Helper method per generare l'HTML delle statistiche."""
        html_sections = []
        
        # Informazioni Dataset
        html_sections.append(f"""
        <div class="stats-section">
            <h4>Informazioni Dataset</h4>
            <ul>
                <li>Numero di valori: <strong>{stats['count']}</strong></li>
            </ul>
        </div>
        """)
        
        # Valori Principali
        html_sections.append(f"""
        <div class="stats-section">
            <h4>Valori Principali</h4>
            <ul>
                <li>Minimo: <strong>{stats['min_max']['min']:.4f}</strong></li>
                <li>Q1 (25° percentile): <strong>{stats['quartili']['Q1']:.4f}</strong></li>
                <li>Mediana (Q2): <strong>{stats['mediana']:.4f}</strong></li>
                <li>Media: <strong>{stats['media']:.4f}</strong></li>
                <li>Q3 (75° percentile): <strong>{stats['quartili']['Q3']:.4f}</strong></li>
                <li>Massimo: <strong>{stats['min_max']['max']:.4f}</strong></li>
            </ul>
        </div>
        """)
        
        # Misure di Dispersione
        html_sections.append(f"""
        <div class="stats-section">
            <h4>Misure di Dispersione</h4>
            <ul>
                <li>Dev. Std. (pop.): <strong>{stats['deviazione_standard_popolazione']:.6f}</strong></li>
                <li>Dev. Std. (camp.): <strong>{stats['deviazione_standard_campione']:.6f}</strong></li>
                <li>Range: <strong>{stats['range']:.4f}</strong></li>
            </ul>
        </div>
        """)
        
        # Altri Indicatori
        moda_format = f"{stats['moda']:.4f}" if isinstance(stats['moda'], (int, float)) else \
                     ", ".join(f"{x:.4f}" for x in stats['moda'])
        
        html_sections.append(f"""
        <div class="stats-section">
            <h4>Altri Indicatori</h4>
            <ul>
                <li>Moda: <strong>{moda_format}</strong></li>
                <li>Varianza (pop.): <strong>{stats['varianza_popolazione']:.6f}</strong></li>
                <li>Varianza (camp.): <strong>{stats['varianza_campione']:.6f}</strong></li>
            </ul>
        </div>
        """)
        
        return "\n".join(html_sections)