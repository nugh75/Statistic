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
    def calcola_tutte_statistiche(numeri: List[float]) -> Dict[str, Union[float, List[float], Dict[str, float]]]:
        """
        Calcola tutte le statistiche disponibili per una lista di numeri.
        
        Args:
            numeri: Lista di numeri
            
        Returns:
            Dict: Dizionario con tutte le statistiche calcolate
        """
        StatisticheCalcolatore.valida_input(numeri)
        return {
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