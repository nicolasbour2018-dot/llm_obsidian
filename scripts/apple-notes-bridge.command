#!/bin/zsh
# Double-clic dans le Finder : lance le pont Apple Notes dans un Terminal.
# Le Terminal doit avoir l'« Accès complet au disque » pour lire la base Notes.
cd "$(dirname "$0")/.." || exit 1
exec /usr/bin/python3 apple_notes_bridge.py serve
