import os
import subprocess


def format_tests_directory():
    tests_dir = "tests"
    if not os.path.exists(tests_dir):
        print(f"Le dossier '{tests_dir}' n'existe pas.")
        return

    print(
        "Formatage des fichiers Python dans 'tests/' avec autopep8 (max-line-length = 120)...\n"
    )

    try:
        subprocess.run(
            [
                "autopep8",
                tests_dir,
                "--in-place",
                "--recursive",
                "--max-line-length",
                "120",
            ],
            check=True,
        )
        print("✅ Formatage terminé avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du formatage : {e}")


if __name__ == "__main__":
    format_tests_directory()
