#!/usr/bin/env python
"""
Import the dataset and save it in a csv and xlsx file. 
After, load the dataset from the csv file and vectorize the texts.
And finally, train the model and test it.
"""
import os
import pandas as pd


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
