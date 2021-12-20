import pandas as pd
import os

dataset_path = '.\COVID-19_Radiography_Dataset'
files = ['COVID.metadata.xlsx', 'Lung_Opacity.metadata.xlsx',
         'Normal.metadata.xlsx', 'Viral Pneumonia.metadata.xlsx']
disease_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']


def read_dataset_concat_label(filename, label, label_name):
    df = pd.read_excel(os.path.join(
        dataset_path, filename), sheet_name='Sheet1')
    df['label'] = pd.Series(len(df) * [label])
    df['label_name'] = pd.Series(len(df) * [label_name])
    df['FILE NAME'] = df['FILE NAME'].apply(lambda x: x + '.png')

    return df


all_data = pd.DataFrame(
    columns=['FILE NAME', 'FORMAT', 'SIZE', 'URL', 'label', 'label_name'])

for ind, f in enumerate(files):

    df = read_dataset_concat_label(f, ind, disease_names[ind])
    all_data = pd.concat([all_data, df], ignore_index=True)

all_data.to_csv('COVID-19_Radiography_Dataset_metadata.csv', index=False)
