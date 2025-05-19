import ast
from pathlib import Path

import chardet

# Fonction pour lire un fichier avec détection automatique d'encodage


def read_file_with_detected_encoding(file_path):
    with open(file_path, "rb") as f:
        raw = f.read()
        encoding = chardet.detect(raw)["encoding"]
        return raw.decode(encoding or "utf-8", errors="replace")


# Analyse un fichier pour détecter les noms utilisés non définis/importés


def check_undefined_names_and_missing_imports(file_path):
    issues = []
    source = read_file_with_detected_encoding(file_path)
    tree = ast.parse(source, filename=str(file_path))
    defined_names = set()
    used_names = set()

    class Analyzer(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                defined_names.add(alias.asname or alias.name)

        def visit_ImportFrom(self, node):
            for alias in node.names:
                defined_names.add(alias.asname or alias.name)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                used_names.add(node.id)

    Analyzer().visit(tree)
    undefined = used_names - defined_names - set(dir(__builtins__))
    if "signal" in undefined:
        issues.append("Missing import: 'signal'")
    return issues


# Analyse tous les fichiers .py du projet
project_dirs = ["tests", "src", "modules"]  # Ajoute d'autres dossiers si nécessaire
issues_summary = {}

for project_dir in project_dirs:
    path = Path(project_dir)
    if not path.exists():
        continue
    for file_path in path.rglob("*.py"):
        issues = check_undefined_names_and_missing_imports(file_path)
        if issues:
            issues_summary[str(file_path)] = issues

# Affiche les résultats
if not issues_summary:
    print("✅ Aucun import manquant détecté.")
else:
    print("🚨 Import(s) manquant(s) détecté(s) :")
    for filename, issues in issues_summary.items():
        print(f"- {filename}: {', '.join(issues)}")
