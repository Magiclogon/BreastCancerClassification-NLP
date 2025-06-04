import os
import pandas as pd
import re
from googletrans import Translator

folder_path = "Dataset"
dataset_path = "./Preprocessing/dataset_tabular.csv"
pattern = r"BI-RADS\s*-?\s*(\d+[A-Za-z]?)"
translator = Translator()

def extract_result(text):
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        print(matches[-1])
        return matches[-1]
    else:
        return "NONE"

def translate_text(text):
    try:
        translation = translator.translate(text, src="pt", dest="en")
        return translation.text
    except Exception as e:
        print(f"Error translating: {e}")
        return text

def removing_bi_rads(text):
    pattern = r"-?\s*Bi\s*-\s*Rads\s*-\s*\d\w?.?$"
    sub_string = re.findall(pattern , text, re.IGNORECASE)
    sub_string = sub_string[-1] if len(sub_string) else ""
    return text.replace(sub_string,"")

data = []
i = 0
report = ""
for file in os.listdir(folder_path):
    path = os.path.join(folder_path, file)

    with open(path, "r") as f:
        report = f.read()

    birads = extract_result(report)

    if birads == "NONE":
        continue

    translated_report = translate_text(report)
    translated_report = removing_bi_rads(translated_report) 
    data.append({"filename": file, "original_text": report, "translated_text": translated_report, "BI-RADS": birads})
    i += 1
    print(i)


df = pd.DataFrame(data)

df.to_csv(dataset_path, index=False, encoding="utf-8")

print("Formatting done!")

