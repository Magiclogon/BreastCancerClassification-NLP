import os
import pandas as pd
import re
from googletrans import Translator

# Dossier contenant les fichiers de rapports médicaux
folder_path = "Dataset"

# Chemin du fichier CSV de sortie
dataset_path = "./Preprocessing/dataset.csv"

# Motif regex pour extraire les scores BI-RADS du texte
pattern = r"BI-RADS\s*-?\s*(\d\w?)"

# Initialisation du traducteur Google Translate
translator = Translator()

# Fonction pour extraire le score BI-RADS depuis un texte
def extract_result(text):
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        print(matches[-1])  # Affiche le score BI-RADS trouvé
        return matches[-1]
    else:
        return "NONE"  # Aucun score trouvé

# Fonction pour traduire un texte du portugais vers l'anglais
def translate_text(text):
    try:
        translation = translator.translate(text, src="pt", dest="en")
        return translation.text  # Texte traduit
    except Exception as e:
        print(f"Error translating: {e}")
        return text  # Retourne le texte original en cas d'erreur

# Fonction pour supprimer les lignes contenant le motif "BI-RADS-X"
def removing_bi_rads(text):
    pattern = r"-?\s*Bi\s*-\s*Rads\s*-\s*\d\w?.?"
    regex = re.compile(pattern, re.IGNORECASE)

    # Filtrer les lignes qui ne contiennent PAS le motif
    lignes_filtrees = [ligne for ligne in text.splitlines() if not regex.search(ligne)]

    # Recomposer le texte filtré
    texte_filtré = "\n".join(lignes_filtrees)
    return texte_filtré


# Liste pour stocker les données finales
data = []
i = 0  # Compteur de fichiers traités

# Parcourt chaque fichier dans le dossier
for file in os.listdir(folder_path):
    path = os.path.join(folder_path, file)

    # Lit le contenu complet du fichier
    with open(path, "r") as f:
        report = f.read()

    # Extraction du score BI-RADS
    birads = extract_result(report)

    # Ignore les rapports sans BI-RADS
    if birads == "NONE":
        continue

    # Traduction du rapport en anglais
    translated_report = translate_text(report)

    # Suppression de la ligne contenant "BI-RADS-X" après traduction
    translated_report = removing_bi_rads(translated_report)

    # Ajout des données collectées dans la liste
    data.append({
        "filename": file,
        "original_text": report,
        "translated_text": translated_report,
        "BI-RADS": birads
    })

    i += 1
    print(i)  # Affiche le nombre de fichiers traités

# Création d'un DataFrame Pandas à partir des données
df = pd.DataFrame(data)

# Sauvegarde du DataFrame dans un fichier CSV
df.to_csv(dataset_path, index=False, encoding="utf-8")

print("Formatting done!")
