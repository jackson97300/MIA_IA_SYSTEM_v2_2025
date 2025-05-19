import os
import re
import shutil

# Chemins
source_file = "D:\\MIA_IA_SYSTEM_v2_2025\\src\\features\\features_audit.py"
cleaned_file = "D:\\MIA_IA_SYSTEM_v2_2025\\src\\features\\features_audit_clean.py"
backup_file = source_file + ".bak"

# Remplacements des caractères typographiques (si nécessaire)
replacements = {
    "\x92": "'",  # apostrophe typographique
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
}

try:
    # 🔄 Lecture initiale en latin1
    with open(source_file, "r", encoding="latin1") as f:
        content = f.read()
    print("✅ Fichier lu avec succès en latin1")

    # 🔍 Analyse rapide autour de la position 61
    if len(content) >= 70:
        print("🔍 Caractères autour de la position 61 (50-70):", content[50:70])

    # Détection des guillemets typographiques
    if any(c in content for c in ["’", "‘", "“", "”"]):
        print("⚠️ Guillemets typographiques détectés")
    else:
        print("✅ Aucun guillemet typographique détecté")

    # Détection des caractères non-ASCII
    non_ascii = re.findall(r"[^\x20-\x7E\n\t]", content)
    if non_ascii:
        print(f"⚠️ Caractères non-ASCII détectés : {non_ascii}")
    else:
        print("✅ Aucun caractère non-ASCII suspect détecté")

    # 🔧 Nettoyage des caractères typographiques
    for old, new in replacements.items():
        content = content.replace(old, new)

    # 💾 Sauvegarde UTF-8 temporaire
    with open(cleaned_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Fichier nettoyé sauvegardé sous : {os.path.abspath(cleaned_file)}")

    # 🔁 Remplacement automatique avec backup
    shutil.move(source_file, backup_file)
    shutil.move(cleaned_file, source_file)
    print(
        f"♻️ Remplacement effectué. L'ancien fichier a été sauvegardé sous : {backup_file}"
    )

except Exception as e:
    print(f"❌ Erreur : {e}")
