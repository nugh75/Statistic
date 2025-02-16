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
    def calcola_correlazioni(serie_dati: Dict[str, List[float]]) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
        """
        Calcola le correlazioni tra tutte le serie di dati utilizzando numpy.
        Gestisce serie di lunghezze diverse troncando alla lunghezza minima comune.
        Calcola anche la significatività statistica (p-value) per ogni correlazione.
        
        Args:
            serie_dati: Dizionario con nome serie come chiave e lista di valori come valore
            
        Returns:
            Dict: Matrice di correlazione con coefficienti e p-values
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
                if col1 != col2:
                    corr, p_value = stats.pearsonr(df[col1], df[col2])
                    result[col1][col2] = {
                        'coefficiente': corr,
                        'p_value': p_value
                    }
                else:
                    result[col1][col2] = {
                        'coefficiente': 1.0,
                        'p_value': 0.0
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