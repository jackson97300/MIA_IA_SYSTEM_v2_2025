import subprocess


def run_isort():
    print("🔧 Exécution de isort...")
    result = subprocess.run(["isort", ".", "--diff"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Aucun fichier à corriger avec isort.")
    else:
        print("⚠️ Fichiers à reformater avec isort :")
        print(result.stdout)

    # Appliquer les changements automatiquement
    subprocess.run(["isort", "."], check=True)
    print("✅ isort appliqué avec succès.")


def list_modified_files():
    print("\n📄 Fichiers modifiés après isort :")
    result = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True
    )
    print(result.stdout if result.stdout else "✅ Aucun fichier modifié détecté.")


if __name__ == "__main__":
    if not shutil.which("isort"):
        print("❌ isort n'est pas installé. Installe-le avec : pip install isort")
    else:
        run_isort()
        list_modified_files()
