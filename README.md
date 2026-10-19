# 🧠 Second Cerveau — RAG Mistral sur Obsidian

Interface Streamlit qui discute avec le coffre Obsidian « Nico_2.0 » (structure
PARA) grâce au paquet local `obsidian-rag` et à l'API Mistral.

## Fonctionnalités

- **💬 Chat** : réponses sourcées (`[S1]`…) avec extraits, score et lien
  « Ouvrir dans Obsidian » ; avertissement si les citations sont douteuses.
  Filtre par dossier PARA et réglage du Top-K dans la barre latérale.
- **🧠 Mémoire LLM** : faits durables relus à chaque question. Bouton
  « Mémoriser » sous une réponse, ou message commençant par `retiens`,
  `mémorise` ou `/memo`. Éditable dans la barre latérale.
- **💾 Conversations** enregistrées comme notes Obsidian (désactivable).
- **🔎 Recherche** sémantique sans LLM et **🎲 Redécouvrir** une note au
  hasard avec ses notes connexes.
- **🍎 Apple Notes** : import des nouvelles notes dans `Boite de réception/Apple Notes/`
  via un pont local (dédoublonnage exact avec les notes déjà importées par
  `obsidian-importer`, tableaux convertis en Markdown, texte OCR des images et
  scans en callout repliable, liens conservés).
- **🔁 Repli OpenAI** : si Mistral refuse (modèle hors abonnement, quota, panne),
  le chat bascule automatiquement sur OpenAI ; le modèle utilisé s'affiche sous
  chaque réponse.

## Configuration

Tout se règle dans le fichier `.env` local (ignoré par Git, modèle :
`.env.example`) :

- `MISTRAL_API_KEY` : clé Mistral (embeddings + chat).
- `MISTRAL_CHAT_MODEL` : modèle de chat, `mistral-small-latest` par défaut
  (`mistral-large-latest` n'est pas inclus dans tous les abonnements).
- `OPENAI_API_KEY` / `OPENAI_CHAT_MODEL` (`gpt-5-mini` par défaut) : repli
  automatique du chat. `CHAT_PROVIDER=openai` force OpenAI pour le chat. Les
  embeddings restent toujours Mistral (l'index est construit avec `mistral-embed`).
- `OBSIDIAN_VAULT_PATH` : **chemin du coffre sur le Mac** (ex.
  `/Users/nicolasbour/Developer/Nico_2.0`). Il sert uniquement aux volumes
  Docker ; dans le conteneur, le coffre est toujours monté sous `/vault`.
- `OBSIDIAN_VAULT_NAME` : nom du coffre pour les liens `obsidian://`.
- `RAG_EXCLUDE_DIRS` : dossiers exclus de l'index (par défaut les
  conversations enregistrées et `Recently Deleted`).
- `APPLE_NOTES_BRIDGE_TOKEN` : jeton partagé avec le pont Apple Notes.

## Lancement

```bash
mkdir -p "$OBSIDIAN_VAULT_PATH/0 Second Cerveau"   # une seule fois
docker compose up -d --build
```

L'application est alors disponible sur <http://127.0.0.1:8590>.

Le coffre est monté **en lecture seule**, à l'exception de `0 Second Cerveau/`,
seul dossier modifiable par l'IA : il contient `Mémoire LLM.md` et le
sous-dossier `Conversations/`. L'index SQLite est conservé dans un volume Docker.
Seuls les `RAG_TOP_K` passages les plus pertinents (6 par défaut) sont envoyés à
Mistral pour chaque question.

## Pont Apple Notes

Docker n'a pas accès à Apple Notes : un petit pont tourne directement sur le Mac
et écoute sur le port **8591** (joint par le conteneur via
`host.docker.internal`).

Le pont lit lui-même `.env` (`OBSIDIAN_VAULT_PATH`, `APPLE_NOTES_BRIDGE_TOKEN`) :

```bash
python3 apple_notes_bridge.py sync --dry-run   # aperçu, sans rien écrire
python3 apple_notes_bridge.py serve            # à laisser tourner
```

Ou double-cliquer sur `scripts/apple-notes-bridge.command` dans le Finder.

Le pont lit une **copie** de la base `NoteStore.sqlite` de Notes.app (la base
d'origine n'est jamais modifiée). Ses identifiants sont les UUID que
`obsidian-importer` a écrits dans les `apple-notes-id` : les notes déjà présentes
dans le coffre, même rangées ailleurs, ne sont jamais réimportées ni écrasées.
Une note importée n'est mise à jour que si elle a changé dans Notes **et** n'a pas
été modifiée dans Obsidian. Les notes verrouillées et la corbeille sont ignorées ;
images, PDF et audio restent dans Notes (un marqueur indique leur emplacement).

Si la lecture est refusée, accorder **Accès complet au disque** au Terminal
(Réglages Système → Confidentialité et sécurité). En secours,
`APPLE_NOTES_EXPORT=jxa` passe par AppleScript (lent, sans dédoublonnage exact).

Les boutons « Aperçu » et « Importer dans Obsidian » de la barre latérale sont
actifs dès que le pont répond ; un import réindexe immédiatement les notes.

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
