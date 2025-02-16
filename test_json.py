import json

def test_json_serialization(data):
    """Test di serializzazione JSON con dati di esempio."""
    try:
        print("Dati originali:", data)
        json_str = json.dumps(data)
        print("JSON serializzato:", json_str)
        parsed = json.loads(json_str)
        print("JSON deserializzato:", parsed)
        return True
    except Exception as e:
        print("Errore durante la serializzazione:", str(e))
        return False

# Test con dati di esempio
test_data = {
    'media': 25.5,
    'mediana': 24.0,
    'moda': [22.0, 25.0],
    'deviazione_standard': 2.5,
    'quartili': {
        'Q1': 22.0,
        'Q2': 24.0,
        'Q3': 26.0
    }
}

if __name__ == '__main__':
    test_json_serialization(test_data)