"""Interface terminal dont stdout reste strictement JSON."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence, TextIO

from .errors import RAGError
from .rag import ObsidianRAG
from .settings import Settings


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise RAGError("cli_usage_error", f"Arguments invalides : {message}", exit_code=2)


def _parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser(prog="obsidian-rag")
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=_JSONArgumentParser
    )
    ask = subparsers.add_parser("ask", help="Poser une question ponctuelle")
    ask.add_argument("question")
    subparsers.add_parser("chat", help="Démarrer une session interactive")
    index = subparsers.add_parser("index", help="Actualiser l'index")
    index.add_argument("--rebuild", action="store_true", help="Reconstruire tout l'index")
    return parser


def _write_json(stream: TextIO, payload: Any, pretty: bool) -> None:
    if pretty:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    else:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
    stream.write("\n")
    stream.flush()


def _chat(rag: ObsidianRAG, pretty: bool, stdin: TextIO, stdout: TextIO, stderr: TextIO) -> None:
    history: list[tuple[str, str]] = []
    turn = 0
    print("Chat prêt. /reset vide la mémoire, /quit ferme la session.", file=stderr)
    while True:
        print("> ", end="", file=stderr, flush=True)
        line = stdin.readline()
        if line == "":
            break
        question = line.strip()
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/reset":
            history.clear()
            turn = 0
            print("Mémoire de conversation réinitialisée.", file=stderr)
            continue
        previous_count = len(history[-6:])
        print("Actualisation de l'index et génération…", file=stderr)
        result = rag.ask(question, history=history)
        turn += 1
        result.diagnostics["session"] = {
            "mode": "interactive",
            "turn": turn,
            "history_exchanges": previous_count,
        }
        _write_json(stdout, result.to_dict(), pretty)
        history.append((question, result.answer))
        history[:] = history[-6:]


def run(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    settings: Settings | None = None,
    rag_factory: Any = ObsidianRAG,
) -> int:
    args_list = list(argv if argv is not None else sys.argv[1:])
    pretty = False
    while "--pretty" in args_list:
        pretty = True
        args_list.remove("--pretty")
    try:
        args = _parser().parse_args(args_list)
        loaded_settings = settings or Settings.from_env()
        with rag_factory(loaded_settings) as rag:
            if args.command == "index":
                print("Actualisation de l'index…", file=stderr)
                update = rag.refresh_index(force=args.rebuild)
                _write_json(stdout, {"index_update": update.to_dict()}, pretty)
            elif args.command == "ask":
                print("Actualisation de l'index et génération…", file=stderr)
                result = rag.ask(args.question)
                _write_json(stdout, result.to_dict(), pretty)
            else:
                _chat(rag, pretty, stdin, stdout, stderr)
        return 0
    except RAGError as exc:
        _write_json(stderr, {"error": {"code": exc.code, "message": exc.message}}, pretty)
        return exc.exit_code
    except KeyboardInterrupt:
        _write_json(stderr, {"error": {"code": "interrupted", "message": "Opération interrompue."}}, pretty)
        return 130
    except Exception:
        # Ne sérialise jamais l'exception brute : certains SDK incluent des en-têtes
        # ou paramètres sensibles dans leur représentation textuelle.
        _write_json(
            stderr,
            {"error": {"code": "internal_error", "message": "Une erreur interne est survenue."}},
            pretty,
        )
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":  # pragma: no cover
    main()
