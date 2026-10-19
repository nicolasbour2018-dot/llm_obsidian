FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Installe d'abord les dépendances pour conserver cette couche entre deux
# modifications du seul code applicatif.
COPY vendor/obsidian-rag /opt/obsidian-rag
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py chat_backend.py brain_vault.py apple_notes_client.py apple_notes_bridge.py ./
COPY tests ./tests

EXPOSE 8501

# Le point de santé natif de Streamlit permet à Docker de détecter un blocage.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
