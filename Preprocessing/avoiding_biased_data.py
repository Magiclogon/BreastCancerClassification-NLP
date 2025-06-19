import pandas as pd
from sklearn.utils import resample
from datasets import Dataset

dataset_path = "./Preprocessing/dataset_tabular.csv"
df = pd.read_csv(dataset_path)

target_size = 250
balanced_data = []

for label in df['BI-RADS'].unique():
    class_df = df[df['BI-RADS'] == label]

    upsampled = resample(
        class_df,
        replace=True,
        n_samples=target_size,
        random_state=42
    )

    balanced_data.append(upsampled)

balanced_df = pd.concat(balanced_data).reset_index(drop=True)

