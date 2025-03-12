import pandas as pd
from sklearn.utils import resample
from datasets import Dataset


"""
    The data we have is biased:
    
                    BI-RADS
                    2     56
                    5     24
                    4c    10
                    3      7
                    4a     4
                    6      3
                    4b     2
                    4      1
                    Name: count, dtype: int64         


    so we adopted the following approach where we choose to duplicate underrepresented samples using sklearn's resample



"""


dataset_path = ".\\Preprocessing\\dataset.csv"
df = pd.read_csv(dataset_path)

majority_class = df[df['BI-RADS'] == '2']
target_size = len(majority_class)  # 56

balanced_data = [majority_class]

for label in df['BI-RADS'].unique():
    if label != "2":  
        minority_class = df[df['BI-RADS'] == label]
        upsampled = resample(
            minority_class, replace=True, n_samples=target_size, random_state=42
        )
        balanced_data.append(upsampled)


balanced_df = pd.concat(balanced_data)

