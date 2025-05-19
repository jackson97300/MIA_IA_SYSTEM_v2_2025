import subprocess
from pathlib import Path

# Dossier à formater : ici le dossier courant
TARGET_DIR = Path(".")


def run_autopep8(path: Path):
    try:
        print("🧽 Application de autopep8...")
        subprocess.run(
            [
                "autopep8",
                "--in-place",
                "--recursive",
                "--max-line-length=120",
                str(path),
            ],
            check=True,
        )
    except Exception as e:
        print("❌ autopep8 erreur:", e)


def run_ruff(path: Path):
    try:
        print("🔧 Application de ruff...")
        subprocess.run(["ruff", "check", "--fix", str(path)], check=True)
    except Exception as e:
        print("❌ ruff erreur:", e)


def run_black(path: Path):
    try:
        print("🖤 Application de black...")
        subprocess.run(["black", str(path)], check=True)
    except Exception as e:
        print("❌ black erreur:", e)


if __name__ == "__main__":
    run_autopep8(TARGET_DIR)
    run_ruff(TARGET_DIR)
    run_black(TARGET_DIR)
    print("🎉 Formatage automatique terminé.")
