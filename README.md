# Obsidian RAG — interface Streamlit

Interface de chat minimale reposant sur le paquet local `obsidian-rag`. Le coffre
Obsidian est monté en lecture seule et l'index SQLite est conservé dans un volume
Docker.

## Lancement local

```bash
docker compose up -d --build
```

L'application est alors disponible sur <http://127.0.0.1:8590>.

Les identifiants Mistral se configurent dans le fichier `.env` local, ignoré par
Git. Le moteur indexe tout le coffre mais ne transmet à Mistral que les
`RAG_TOP_K` passages les plus pertinents pour chaque question (6 par défaut).

## Tunnel Cloudflare temporaire

```bash
cloudflared tunnel --url http://127.0.0.1:8590
```

Le lien `trycloudflare.com` est public et cesse de fonctionner à l'arrêt de la
commande.

## Vérifications

```bash
docker compose exec app python -m unittest discover -s tests -v
curl --fail http://127.0.0.1:8590/_stcore/health
```
