import ast
import traceback
from pathlib import Path


def show_error_context(file_path, line_number, context=2):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
        start = max(0, line_number - context - 1)
        end = min(len(lines), line_number + context)
        snippet = "".join(
            f"{i + 1:4d}: {'>> ' if i + 1 == line_number else '   '}{lines[i]}"
            for i in range(start, end)
        )
        return snippet


def diagnose_syntax_errors(root_dir=".", exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = {".venv", "__pycache__"}

    print("📂 Analyse des fichiers Python...\n")
    for path in Path(root_dir).rglob("*.py"):
        if any(ex in str(path) for ex in exclude_dirs):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=str(path))
        except SyntaxError as e:
            print(f"❌ Fichier: {path}")
            print(f"   ➤ Ligne {e.lineno}, Colonne {e.offset}: {e.msg}")
            print("   Contexte :")
            print(show_error_context(path, e.lineno))
            print("-" * 80)
        except Exception as e:
            print(f"⚠️ Erreur non-syntaxique dans {path}: {e}")
            print(traceback.format_exc())
            print("-" * 80)


if __name__ == "__main__":
    diagnose_syntax_errors(root_dir=".")
