#!/usr/bin/env python
"""
Import the dataset and save it in a csv and xlsx file. 
After, load the dataset from the csv file and vectorize the texts.
And finally, train the model and test it.
"""
import os
import pandas as pd

# Classes
from classes.StackedCLSModel import StackedCLSModel


def main():
    # Load data
    dataset_path = os.environ.get('dataset_path')
    dataframe_path = os.environ.get('dataframe_path')

    # Dataframe from the json file
    df = pd.read_json(dataset_path)

    # Save the dataframe in a csv and xlsx file.
    SaveDataSet(df)

    # Get the dataframe from the csv file
    dataframe = pd.read_csv(os.path.join(dataframe_path, 'dataframe.csv'))
    # datframe2 = pd.read_excel(os.path.join(dataframe_path, 'dataframe.xlsx'))

    # Data to test
    df = df.iloc[:8, :]
    dataframe = dataframe.iloc[:8, :]

    # Model
    model = StackedCLSModel()

    # Vectorize texts for all rows from the dataframe
    df['text_vec'] = df.apply(lambda r: model.vectorize_text(
        r['pair'][0], r['pair'][1], 512), axis=1)
    dataframe['text_vec'] = df.apply(lambda r: model.vectorize_text(
        r['pair'][0], r['pair'][1], 512), axis=1)

    # DATA FOR TRAINING AND TESTING - Specifies data ratio for training and testing
    # seleccionar aleatoriamente todas las filas en el DataFrame
    df = df.sample(frac=1)
    # especifica la proporción de los datos que se usarán para el entrenamiento
    train_portion = float(os.environ.get('train_portion'))
    # calcula el índice de la fila en la que dividir los datos en subconjuntos de entrenamiento y prueba
    split_point = int(train_portion*len(df))
    # asigna las filas anteriores al punto de división train_data y las filas posteriores al punto de división a test_data
    train_data, test_data = df[:split_point].reset_index(
        drop=True), df[split_point:].reset_index(drop=True)
    # train_data = train_data.reset_index(drop=True)
    print({
        'train_data': len(train_data),
        'test_data': len(test_data)
    })

    train_set, test_set = MyDataset(train_data), MyDataset(test_data)


def SaveDataSet(df):
    """
    Save the dataset in a csv and xlsx file.
    """
    dataframe_path = os.environ.get('dataframe_path')
    # _continue = os.environ.get('save_dataframe') # The variable 'save_dataframe' force to save the dataframe

    csv_path = os.path.join(dataframe_path, 'dataframe.csv')
    xlsx_path = os.path.join(dataframe_path, 'dataframe.xlsx')

    if os.path.exists(csv_path) or os.path.exists(xlsx_path):
        # print("Folder already exists")
        return

    # Order DataFrame
    keys = list(df.keys())
    df_to_file = pd.DataFrame([{key: df[key][pos]
                                for key in keys} for pos in df[keys[0]].keys()])

    # Export DataFrame to CSV file
    df_to_file.to_csv(csv_path,
                      encoding='utf-8-sig', index=False)

    # Export DataFrame to XLSX file
    df_to_file.to_excel(xlsx_path,
                        encoding='utf-8-sig', index=False)
