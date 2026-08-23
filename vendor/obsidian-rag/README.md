# Obsidian RAG avec Mistral Large

Ce paquet lit un coffre Obsidian **en lecture seule**, indexe uniquement ses notes Markdown dans SQLite, puis répond à partir des passages retrouvés avec Mistral Large. Obsidian n'a pas besoin d'être ouvert et aucun plugin n'est nécessaire.

## Installation

Python 3.12 ou 3.13 est requis. Avec Conda :

```bash
conda create -n obsidian-rag python=3.13 -y
conda activate obsidian-rag
python -m pip install -e '.[dev]'
cp .env.example .env
```

Renseignez ensuite `MISTRAL_API_KEY` et le chemin absolu `OBSIDIAN_VAULT_PATH` dans `.env`. Le fichier `.env` et l'index local sont ignorés par Git. Les variables déjà définies dans le système ont priorité sur `.env`.

Pour vérifier le SDK Mistral installé :

```bash
python -c "from mistralai.client import Mistral; print('SDK Mistral installé')"
```

## Utilisation

```bash
obsidian-rag index --rebuild
obsidian-rag ask "Que disent mes notes sur ce sujet ?"
obsidian-rag ask --pretty "Résume le projet Alpha"
obsidian-rag chat
```

Dans le chat, `/reset` vide les six échanges mémorisés et `/quit` ferme la session. Les résultats sont émis en JSON sur `stdout`; les invites, progrès et erreurs vont sur `stderr`. `--pretty` peut être placé avant ou après la sous-commande.

Le cœur est également importable :

```python
from obsidian_rag import ObsidianRAG, Settings

settings = Settings.from_env()
with ObsidianRAG(settings) as rag:
    update = rag.refresh_index()
    result = rag.ask("Ma question")
    print(result.to_dict())
```

L'index est actualisé avant chaque question : seules les notes ajoutées ou modifiées sont ré-encodées, et les suppressions sont propagées. Un changement de modèle d'embedding ou de version du découpage reconstruit tout l'index.

## Tests

```bash
pytest -m "not live"
```

Le test réel est volontairement optionnel :

```bash
RUN_MISTRAL_LIVE_TEST=1 pytest -m live
```

Il nécessite une clé valide dans l'environnement ou `.env` et consomme un petit nombre de jetons API.
