# Streamlit minimal « Obsidian RAG »

  ## Résumé

  Créer une application Streamlit affichant uniquement le titre Obsidian RAG, l’historique du chat et le champ de
  saisie. Elle utilisera directement le paquet Python existant obsidian_rag, sans modifier son code, puis sera lancée
  dans Docker et exposée par un Quick Tunnel Cloudflare temporaire.

  ## Changements principaux

  - Copier comme dépendance locale uniquement le pyproject.toml, le README.md et src/obsidian_rag du projet
    d’inférence ; exclure .env, tests, caches et fichiers générés.

  - Appeler l’API publique existante Settings.from_env() puis ObsidianRAG.ask(question, history=...).
  - Conserver l’affichage complet de la conversation dans st.session_state, tout en transmettant seulement les six
    derniers échanges réussis au RAG.

  - Ne rendre aucun panneau, bouton, diagnostic ou détail de source ; afficher les erreurs sûres dans une bulle
    assistant et ne pas les ajouter à l’historique RAG.

  - Sérialiser les appels d’inférence afin d’éviter des écritures SQLite concurrentes lorsque plusieurs visiteurs
    utilisent le lien public.

  ## Docker et exposition

  - Construire une image Python 3.13 avec Streamlit et la dépendance locale obsidian-rag.
  - Monter /Users/nicolasbour/Desktop/Nico_2.0 en lecture seule dans /vault.
  - Stocker l’index SQLite dans un volume Docker persistant pour éviter de recalculer les embeddings après chaque
    redémarrage.

  - Injecter la clé Mistral depuis le .env existant sans la copier dans l’image ; remplacer dans le conteneur
    OBSIDIAN_VAULT_PATH=/vault et RAG_INDEX_PATH=/data/index.sqlite3.

  - Publier Streamlit uniquement sur 127.0.0.1:8502, car 8501 est déjà utilisé.
  - Vérifier le point de santé Streamlit, puis lancer cloudflared tunnel --url http://localhost:8502 et restituer
    l’URL temporaire trycloudflare.com.

  ## Tests et critères d’acceptation

  - Vérifier qu’au chargement seuls le titre, la zone de conversation vide et le champ de saisie sont visibles.
  - Vérifier qu’une question affiche une bulle utilisateur puis la réponse produite par le module d’inférence.
  - Vérifier que les échanges restent isolés par session et que seuls les six derniers sont envoyés au RAG.
  - Vérifier qu’une erreur de configuration ou d’API n’expose ni clé ni exception brute.
  - Vérifier que le coffre est monté en lecture seule, que l’index persiste, que Docker répond sur localhost:8502 et
    que l’URL Cloudflare charge le chat.

  ## Hypothèses retenues

  - Le Quick Tunnel est volontairement public et sans authentification ; il reste disponible uniquement tant que
    cloudflared fonctionne.

  - La clé Mistral configurée dans le projet d’inférence est utilisée sans être copiée ni affichée.
  - Les sources restent exploitées par le RAG pour générer les citations, mais elles ne sont pas présentées séparément
    dans l’interface.

    