import pandas as pd
import os
import re

files = os.listdir("Dataset/")

pattern = r"-?\s*Bi\s*-\s*Rads\s*-?\s*\d\w?.?"
for file in files:
        with open(f"Dataset/{file}","r") as f:
            report = f.read()
        print(report+"--------")
