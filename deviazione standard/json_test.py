import json
from statistiche import StatisticheCalcolatore
import pandas as pd

def test_serializzazione(dati):
    """Testa la serializzazione JSON dei dati statistici."""
    try:
        # Calcola le statistiche
        statistiche = StatisticheCalcolatore.calcola_tutte_statistiche(dati)
        print("\nStatistiche calcolate:")
        for key, value in statistiche.items():
            print(f"{key}: {value} (tipo: {type(value)})")

        # Testa ogni parte separatamente
        print("\nTest serializzazione singoli componenti:")
        
        # Media
        print("\nTest media:")
        json.dumps(float(statistiche['media']))
        print("OK")
        
        # Mediana
        print("\nTest mediana:")
        json.dumps(float(statistiche['mediana']))
        print("OK")
        
        # Moda
        print("\nTest moda:")
        moda = statistiche['moda']
        if isinstance(moda, (list, tuple)):
            moda = [float(x) for x in moda]
        else:
            moda = float(moda)
        json.dumps(moda)
        print("OK")
        
        # Deviazione standard
        print("\nTest deviazione standard:")
        json.dumps(float(statistiche['deviazione_standard_popolazione']))
        print("OK")
        
        # Quartili
        print("\nTest quartili:")
        quartili = {
            'Q1': float(statistiche['quartili']['Q1']),
            'Q2': float(statistiche['quartili']['Q2']),
            'Q3': float(statistiche['quartili']['Q3'])
        }
        json.dumps(quartili)
        print("OK")
        
        # Min/Max
        print("\nTest min/max:")
        min_max = {
            'min': float(statistiche['min_max']['min']),
            'max': float(statistiche['min_max']['max'])
        }
        json.dumps(min_max)
        print("OK")
        
        # Test completo
        print("\nTest serializzazione completa:")
        stats_dict = {
            'media': float(statistiche['media']),
            'mediana': float(statistiche['mediana']),
            'moda': moda,
            'deviazione_standard_popolazione': float(statistiche['deviazione_standard_popolazione']),
            'deviazione_standard_campione': float(statistiche['deviazione_standard_campione']),
            'varianza_popolazione': float(statistiche['varianza_popolazione']),
            'varianza_campione': float(statistiche['varianza_campione']),
            'range': float(statistiche['range']),
            'quartili': quartili,
            'min_max': min_max
        }
        
        json_str = json.dumps(stats_dict)
        print("Serializzazione JSON riuscita:")
        print(json_str)
        return True
        
    except Exception as e:
        print(f"Errore durante il test: {str(e)}")
        return False

if __name__ == '__main__':
    # Dati di test
    test_data = [22.0, 22.0, 25.0, 21.0, 23.0, 23.0, 24.0]
    test_serializzazione(test_data)