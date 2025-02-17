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