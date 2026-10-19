"""Client HTTP (conteneur) du pont Apple Notes qui tourne sur l'hôte macOS.

N'utilise que urllib (bibliothèque standard).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BRIDGE_URL = "http://host.docker.internal:8591"


class BridgeError(Exception):
    """Erreur dont le texte (en français) peut être affiché sans secret."""


def _base_url() -> str:
    """Renvoie l'URL du pont (env APPLE_NOTES_BRIDGE_URL) sans « / » final."""

    return (os.environ.get("APPLE_NOTES_BRIDGE_URL") or DEFAULT_BRIDGE_URL).rstrip("/")


def _decode(raw: bytes) -> Any:
    """Décode une réponse JSON ; None si le contenu n'est pas du JSON."""

    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def bridge_status(timeout: float = 1.5) -> dict | None:
    """Interroge GET /health ; renvoie le JSON ou None si le pont est injoignable."""

    try:
        with urllib.request.urlopen(_base_url() + "/health", timeout=timeout) as response:
            payload = _decode(response.read())
    except Exception:
        # Pont arrêté, hôte injoignable, délai dépassé… : simplement « absent ».
        return None
    return payload if isinstance(payload, dict) else None


def trigger_sync(dry_run: bool, timeout: float = 600) -> dict:
    """Déclenche POST /sync?dry_run=… et renvoie le résumé de synchronisation."""

    token = os.environ.get("APPLE_NOTES_BRIDGE_TOKEN", "")
    if not token:
        raise BridgeError("APPLE_NOTES_BRIDGE_TOKEN n'est pas configuré.")
    url = "%s/sync?dry_run=%d" % (_base_url(), 1 if dry_run else 0)
    request = urllib.request.Request(url, data=b"", method="POST", headers={"X-Bridge-Token": token})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = _decode(response.read())
    except urllib.error.HTTPError as exc:
        # Traduit les codes d'erreur connus en messages lisibles.
        raw = exc.read() or b""
        exc.close()
        if exc.code == 401:
            raise BridgeError("Le pont Apple Notes a refusé le jeton (APPLE_NOTES_BRIDGE_TOKEN).") from None
        if exc.code == 409:
            raise BridgeError("Une synchronisation Apple Notes est déjà en cours.") from None
        body = _decode(raw)
        detail = body.get("error") if isinstance(body, dict) else None
        raise BridgeError(detail or "Le pont Apple Notes a renvoyé l'erreur HTTP %d." % exc.code) from None
    except (urllib.error.URLError, OSError):
        raise BridgeError(
            "Pont Apple Notes injoignable : lancez « python3 apple_notes_bridge.py serve » sur le Mac."
        ) from None
    if not isinstance(payload, dict):
        raise BridgeError("Réponse inattendue du pont Apple Notes.")
    return payload
