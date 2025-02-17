# Usa Python 3.8.10 come base
FROM python:3.8.10

# Imposta la directory di lavoro nel container
WORKDIR /app

# Copia solo il file requirements.txt per installare le dipendenze prima
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il contenuto della cartella nel container
COPY . .

# Esponi la porta 5002 (modifica rispetto a 5000)
EXPOSE 5002

# Comando per avviare Flask
CMD ["python", "app.py"]