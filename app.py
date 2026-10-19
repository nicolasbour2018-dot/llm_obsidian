"""Interface Streamlit du « Second Cerveau » : chat RAG, recherche et mémoire."""

from __future__ import annotations

import os
import random
import re
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote

import streamlit as st

import chat_backend as backend
from chat_backend import ChatBackendError

# Les modules de mémoire et du pont Apple Notes sont facultatifs : leur absence
# désactive la fonctionnalité concernée sans empêcher l'application de démarrer.
try:
    from brain_vault import BrainVault, BrainVaultError
except ImportError:
    BrainVault = None  # type: ignore[assignment,misc]

    class BrainVaultError(Exception):  # type: ignore[no-redef]
        """Remplaçant utilisé quand `brain_vault` est absent."""

try:
    import apple_notes_client
    from apple_notes_client import BridgeError
except ImportError:
    apple_notes_client = None  # type: ignore[assignment]

    class BridgeError(Exception):  # type: ignore[no-redef]
        """Remplaçant utilisé quand `apple_notes_client` est absent."""


FOLDERS = ["0 Second Cerveau", "1 Projet", "2 Domaine", "3 Ressource", "4 Archive", "Boite de réception"]
REDISCOVER_FOLDERS = ("3 Ressource/", "4 Archive/", "2 Domaine/")
VAULT_NAME = os.environ.get("OBSIDIAN_VAULT_NAME", "Nico_2.0")
MAX_EXCHANGES = 6
# « retiens … », « mémorise … » ou « /memo … » enregistrent un fait sans appel RAG.
MEMO_PATTERN = re.compile(r"^\s*(?:retiens|m[ée]morise|/memo)\b[\s:,.-]*(.*)$", re.IGNORECASE | re.DOTALL)


class _Unavailable(Exception):
    """Fonctionnalité désactivée ; le message peut être affiché tel quel."""


# --- Utilitaires -----------------------------------------------------------


def _public_error(exc: Exception) -> str:
    """Ne renvoie le texte d'une erreur que s'il est prévu pour l'affichage."""

    if isinstance(exc, (ChatBackendError, _Unavailable, BrainVaultError, BridgeError)):
        return str(exc)
    return "Une erreur interne est survenue."


def _vault() -> Any:
    """Ouvre la mémoire stockée dans le coffre, ou lève `_Unavailable`."""

    if BrainVault is None:
        raise _Unavailable("Module de mémoire indisponible.")
    try:
        return BrainVault.from_env()
    except BrainVaultError as exc:
        raise _Unavailable(str(exc)) from None
    except Exception:
        raise _Unavailable("Mémoire indisponible.") from None


def _read_memory() -> str:
    """Lit la mémoire LLM ; une mémoire illisible vaut une mémoire vide."""

    try:
        return _vault().read_memory()
    except Exception:
        return ""


def _obsidian_link(path: str) -> str:
    """Construit un lien `obsidian://` ouvrant la note dans l'application."""

    file = path[:-3] if path.endswith(".md") else path
    return f"obsidian://open?vault={quote(VAULT_NAME, safe='')}&file={quote(file, safe='')}"


def _folder(path: str) -> str:
    """Renvoie le dossier d'une note, relatif à la racine du coffre."""

    parent = str(PurePosixPath(path).parent)
    return "racine" if parent == "." else parent


def _label(item: Any) -> str:
    """Rend lisible un élément renvoyé par le pont (texte ou dictionnaire)."""

    if isinstance(item, dict):
        return str(item.get("title") or item.get("path") or item.get("name") or item)
    return str(item)


def _render_source(source: dict, label: str | None = None) -> None:
    """Affiche une source : titre, dossier, score, lien Obsidian et extrait."""

    title = source.get("title") or source.get("path", "")
    if source.get("heading"):
        title = f"{title} › {source['heading']}"
    head = f"**[{label}] {title}**" if label else f"**{title}**"
    path = source.get("path", "")
    parts = [head, _folder(path), f"score {float(source.get('score') or 0):.2f}"]
    parts.append(f"[Ouvrir dans Obsidian]({_obsidian_link(path)})")
    st.markdown(" · ".join(parts))
    if source.get("excerpt"):
        st.caption(source["excerpt"])


def _render_sources(sources: list[dict]) -> None:
    """Affiche une liste de sources numérotées."""

    for position, source in enumerate(sources, start=1):
        _render_source(source, source.get("id") or f"S{position}")


# --- État de session --------------------------------------------------------


def _reset_conversation() -> None:
    """Démarre une conversation vierge avec un nouvel identifiant."""

    # `messages` pilote l'affichage, tandis que `exchanges` ne conserve que les
    # tours réussis utilisés comme contexte par le moteur RAG.
    st.session_state.messages = []
    st.session_state.exchanges = []
    st.session_state.related = {}
    st.session_state.conv_id = uuid.uuid4().hex
    st.session_state.conv_note = None
    st.session_state.started_at = datetime.now()


def _init_state() -> None:
    """Initialise l'état de session au premier affichage."""

    if "messages" not in st.session_state:
        _reset_conversation()
    if "memory_text" not in st.session_state or st.session_state.get("memory_dirty"):
        st.session_state.memory_text = _read_memory()
        st.session_state.memory_dirty = False


# --- Barre latérale ---------------------------------------------------------


@st.cache_data(ttl=60, show_spinner=False)
def _cached_stats() -> dict:
    """Statistiques d'index, mises en cache pour éviter d'ouvrir l'index à chaque clic."""

    return backend.stats()


@st.cache_data(ttl=15, show_spinner=False)
def _cached_bridge_status() -> dict | None:
    """État du pont Apple Notes ; `None` signifie injoignable."""

    if apple_notes_client is None:
        return None
    return apple_notes_client.bridge_status()


def _sidebar_index() -> None:
    """Statistiques de l'index et bouton de réindexation."""

    st.subheader("📊 Index")
    try:
        stats = _cached_stats()
        left, right = st.columns(2)
        left.metric("Notes", stats.get("notes", 0))
        right.metric("Passages", stats.get("chunks", 0))
        folders = stats.get("folders") or {}
        if folders:
            st.caption(" · ".join(f"{name} : {count}" for name, count in sorted(folders.items())))
    except Exception as exc:
        st.warning(f"Index indisponible : {_public_error(exc)}")

    if st.button("🔄 Réindexer", use_container_width=True):
        try:
            with st.spinner("Réindexation en cours…"):
                update = backend.reindex()
            st.success(
                f"{update['added']} ajoutée(s), {update['updated']} mise(s) à jour, "
                f"{update['deleted']} supprimée(s)."
            )
            _cached_stats.clear()
        except Exception as exc:
            st.error(_public_error(exc))


def _sidebar_filters() -> None:
    """Filtres de recherche partagés par le chat et la recherche."""

    st.subheader("🎛️ Filtres")
    st.multiselect("Dossiers", FOLDERS, key="folders", placeholder="Tout le coffre")
    st.slider("Top-K", min_value=3, max_value=12, value=6, key="top_k")
    st.toggle("💾 Enregistrer les conversations dans Obsidian", value=True, key="save_conversations")


def _sidebar_memory() -> None:
    """Édition directe de la mémoire persistante du LLM."""

    with st.expander("🧠 Mémoire LLM"):
        if BrainVault is None:
            st.info("Module de mémoire indisponible.")
            return
        st.text_area("Faits retenus sur toi", key="memory_text", height=220)
        if st.button("Enregistrer", key="save_memory"):
            try:
                _vault().write_memory(st.session_state.memory_text)
                st.success("Mémoire enregistrée.")
            except Exception as exc:
                st.error(_public_error(exc))


def _sidebar_apple_notes() -> None:
    """Import d'Apple Notes via le pont local tournant sur le Mac."""

    st.subheader("🍎 Apple Notes")
    try:
        status = _cached_bridge_status()
    except Exception:
        status = None
    if status is None:
        st.button("Importer dans Obsidian", disabled=True, use_container_width=True)
        st.caption("Pont injoignable. Sur le Mac, lance : `python3 apple_notes_bridge.py serve`")
        return

    preview, run = st.columns(2)
    if preview.button("Aperçu", use_container_width=True):
        try:
            with st.spinner("Lecture d'Apple Notes…"):
                report = apple_notes_client.trigger_sync(dry_run=True)
            st.info(
                f"{len(report.get('created', []))} note(s) à créer, "
                f"{len(report.get('updated', []))} à mettre à jour "
                f"(sur {report.get('total', 0)})."
            )
        except Exception as exc:
            st.error(_public_error(exc))
    if run.button("Importer dans Obsidian", use_container_width=True):
        try:
            with st.spinner("Import en cours…"):
                report = apple_notes_client.trigger_sync(dry_run=False)
            _show_import_report(report)
            # Indexe tout de suite les notes importées pour qu'elles soient
            # cherchables sans attendre la prochaine question.
            if report.get("created") or report.get("updated"):
                with st.spinner("Indexation des nouvelles notes…"):
                    backend.reindex()
            _cached_stats.clear()
        except Exception as exc:
            st.error(_public_error(exc))


def _show_import_report(report: dict) -> None:
    """Résume un import Apple Notes et liste les notes créées."""

    created = report.get("created") or []
    errors = report.get("errors") or []
    st.success(
        f"{len(created)} créée(s), {len(report.get('updated') or [])} mise(s) à jour, "
        f"{report.get('skipped', 0)} inchangée(s), {len(errors)} erreur(s)."
    )
    if created:
        st.markdown("\n".join(f"- {_label(item)}" for item in created))
    if errors:
        st.warning("\n".join(f"- {_label(item)}" for item in errors))


def _render_sidebar() -> None:
    """Assemble la barre latérale."""

    with st.sidebar:
        st.button("➕ Nouvelle conversation", on_click=_reset_conversation, use_container_width=True)
        _sidebar_index()
        _sidebar_filters()
        _sidebar_memory()
        _sidebar_apple_notes()


# --- Chat -------------------------------------------------------------------


def _path_prefixes() -> list[str] | None:
    """Dossiers sélectionnés ; aucune sélection signifie tout le coffre."""

    return list(st.session_state.get("folders") or []) or None


def _save_conversation() -> None:
    """Enregistre la conversation dans le coffre si l'option est active."""

    if not st.session_state.get("save_conversations", True):
        return
    messages = [
        {
            "role": message["role"],
            "content": message["content"],
            "sources": [{"path": s.get("path", ""), "title": s.get("title", "")} for s in message.get("sources", [])],
        }
        for message in st.session_state.messages
        if message.get("kind") != "error"
    ]
    title = next((m["content"] for m in messages if m["role"] == "user"), "Conversation")
    try:
        st.session_state.conv_note = _vault().save_conversation(
            st.session_state.conv_id,
            title[:80],
            messages,
            started_at=st.session_state.started_at,
        )
    except Exception as exc:
        st.toast(f"⚠️ Conversation non enregistrée : {_public_error(exc)}")


def _memorize(question: str, answer: str) -> None:
    """Distille un échange en faits durables et les ajoute à la mémoire."""

    try:
        with st.spinner("Extraction des souvenirs…"):
            facts = backend.distill_memory(question, answer)
        if not facts:
            st.toast("Rien de durable à retenir.")
            return
        count = _vault().append_memory(facts, link=st.session_state.conv_note)
        st.session_state.memory_dirty = True
        st.toast(f"🧠 {count} souvenir(s) ajouté(s).")
    except Exception as exc:
        st.error(_public_error(exc))


def _load_related(index: int, path: str) -> None:
    """Charge les notes connexes d'une source et les garde en session."""

    try:
        st.session_state.related[index] = backend.related(path, k=5)
    except Exception as exc:
        st.session_state.related[index] = _public_error(exc)


def _render_assistant_extras(index: int, message: dict) -> None:
    """Avertissement, sources et actions sous une réponse du RAG."""

    if message.get("warning"):
        st.warning(message["warning"])
    if message.get("model"):
        # Indique notamment quand le repli OpenAI a pris le relais.
        st.caption(f"Modèle : {message['model']}")
    sources = message.get("sources") or []
    if sources:
        with st.expander(f"📚 Sources ({len(sources)})"):
            _render_sources(sources)

    left, right = st.columns(2)
    if left.button("🧠 Mémoriser", key=f"memorize_{index}"):
        _memorize(message.get("question", ""), message["content"])
    if sources and right.button("🔗 Notes connexes", key=f"related_{index}"):
        _load_related(index, sources[0]["path"])

    related = st.session_state.related.get(index)
    if isinstance(related, str):
        st.error(related)
    elif related is not None:
        st.markdown("**🔗 Notes connexes**")
        if not related:
            st.caption("Aucune note connexe trouvée.")
        for source in related:
            _render_source(source)


def _render_message(index: int, message: dict) -> None:
    """Affiche un message de l'historique."""

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("kind") == "answer":
            _render_assistant_extras(index, message)


def _remember(text: str) -> dict:
    """Ajoute directement un fait à la mémoire, sans appel au RAG."""

    if not text.strip():
        return {"role": "assistant", "content": "Rien à retenir : précise le fait après la commande.", "kind": "memo"}
    try:
        _vault().append_memory([text.strip()], link=st.session_state.conv_note)
        st.session_state.memory_dirty = True
        return {"role": "assistant", "content": f"✅ Retenu : {text.strip()}", "kind": "memo"}
    except Exception as exc:
        return {"role": "assistant", "content": f"Erreur : {_public_error(exc)}", "kind": "error"}


def _answer(prompt: str) -> dict:
    """Interroge le RAG ; une erreur produit un message qui n'entre pas dans le contexte."""

    try:
        with st.spinner("Je fouille dans tes notes…"):
            result = backend.ask(
                prompt,
                st.session_state.exchanges[-MAX_EXCHANGES:],
                path_prefixes=_path_prefixes(),
                top_k=st.session_state.get("top_k"),
                memory=_read_memory(),
            )
    except Exception as exc:
        return {"role": "assistant", "content": f"Erreur : {_public_error(exc)}", "kind": "error"}

    # Une erreur affichée ne doit jamais polluer le contexte des questions suivantes :
    # seuls les tours réussis rejoignent `exchanges`.
    st.session_state.exchanges.append((prompt, result.answer))
    st.session_state.exchanges = st.session_state.exchanges[-MAX_EXCHANGES:]
    return {
        "role": "assistant",
        "content": result.answer,
        "kind": "answer",
        "question": prompt,
        "sources": result.sources,
        "warning": result.warning,
        "model": (result.diagnostics.get("models") or {}).get("chat", ""),
    }


def _handle_prompt(prompt: str) -> None:
    """Traite une nouvelle saisie : commande mémoire ou question au RAG."""

    st.session_state.messages.append({"role": "user", "content": prompt, "kind": "user"})
    with st.chat_message("user"):
        st.markdown(prompt)

    memo = MEMO_PATTERN.match(prompt)
    reply = _remember(memo.group(1)) if memo else _answer(prompt)
    st.session_state.messages.append(reply)
    _save_conversation()
    # Le nouveau message est affiché avec ses boutons, sans attendre la prochaine exécution.
    _render_message(len(st.session_state.messages) - 1, reply)


def _chat_tab(prompt: str | None) -> None:
    """Rejoue l'historique puis traite la saisie courante."""

    # Streamlit réexécute ce fichier à chaque interaction : il faut donc rejouer
    # explicitement l'historique pour reconstruire la conversation à l'écran.
    for index, message in enumerate(st.session_state.messages):
        _render_message(index, message)
    if prompt:
        _handle_prompt(prompt)


# --- Recherche et redécouverte ----------------------------------------------


def _search_tab() -> None:
    """Recherche sémantique sans génération par le LLM."""

    query = st.text_input("Rechercher dans tes notes", placeholder="ex. idées de projet sur la photo")
    if not query:
        return
    try:
        with st.spinner("Recherche…"):
            results = backend.search(query, path_prefixes=_path_prefixes(), top_k=st.session_state.get("top_k"))
    except Exception as exc:
        st.error(_public_error(exc))
        return
    if not results:
        st.info("Aucun résultat.")
    for source in results:
        _render_source(source)
        st.divider()


def _pick_random_note() -> None:
    """Tire une note au hasard parmi les ressources, archives et domaines."""

    try:
        notes = [(path, title) for path, title in backend.list_notes() if path.startswith(REDISCOVER_FOLDERS)]
    except Exception as exc:
        st.session_state.rediscover = {"error": _public_error(exc)}
        return
    if not notes:
        st.session_state.rediscover = {"error": "Aucune note à redécouvrir."}
        return
    path, title = random.choice(notes)
    pick: dict = {"path": path, "title": title, "related": []}
    try:
        pick["related"] = backend.related(path, k=5)
    except Exception as exc:
        pick["related_error"] = _public_error(exc)
    st.session_state.rediscover = pick


def _rediscover_tab() -> None:
    """Fait ressurgir une note oubliée et ses voisines."""

    st.button("🎲 Une note au hasard", on_click=_pick_random_note)
    pick = st.session_state.get("rediscover")
    if not pick:
        return
    if "error" in pick:
        st.warning(pick["error"])
        return
    st.markdown(f"### {pick['title']}")
    st.markdown(f"{_folder(pick['path'])} · [Ouvrir dans Obsidian]({_obsidian_link(pick['path'])})")
    if pick.get("related_error"):
        st.warning(pick["related_error"])
    elif pick["related"]:
        st.markdown("**🔗 Notes connexes**")
        for source in pick["related"]:
            _render_source(source)


# --- Page -------------------------------------------------------------------


def main() -> None:
    """Point d'entrée de la page."""

    st.set_page_config(page_title="Second Cerveau", page_icon="🧠", layout="wide")
    st.title("🧠 Second Cerveau")
    st.caption("Discute avec ton coffre Obsidian, retrouve tes notes et nourris ta mémoire.")

    _init_state()
    _render_sidebar()

    # Au niveau racine, le champ de saisie reste épinglé en bas de la page.
    prompt = st.chat_input("Pose ta question… (« retiens … » pour mémoriser)")

    tabs: list[Callable[[], None]] = [lambda: _chat_tab(prompt), _search_tab, _rediscover_tab]
    for container, render in zip(st.tabs(["💬 Chat", "🔎 Recherche", "🎲 Redécouvrir"]), tabs):
        with container:
            render()


main()
