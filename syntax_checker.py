import ast
from pathlib import Path


def check_syntax_errors(directory):
    error_files = []
    py_files = list(Path(directory).rglob("*.py"))

    for file_path in py_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            error_files.append((str(file_path), e.lineno, e.msg))
        except UnicodeDecodeError as e:
            error_files.append((str(file_path), "?", f"Unicode error: {e}"))
        except Exception as e:
            error_files.append((str(file_path), "?", f"Unexpected error: {e}"))

    return error_files


if __name__ == "__main__":
    base_dir = Path(".")  # current directory
    errors = check_syntax_errors(base_dir)
    if not errors:
        print("✅ Tous les fichiers Python sont syntaxiquement valides.")
    else:
        print("❌ Fichiers avec erreurs de syntaxe :\n")
        for path, line, msg in errors:
            print(f"- {path} (ligne {line}) → {msg}")
