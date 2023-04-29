""" MyDataSet Class
    Define the tensor dataset for the model, and structure the data for training and testing.
"""

from sklearn import datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer


class MyDataset(Dataset):             # define una nueva clase MyDataset que hereda de Dataset 
    def __init__(self, dataframe):    # define el constructor  "__init__"  que toma un solo argumento dataframe
        #print(dataframe)
        self.len = len(dataframe)   # calcula la longitud de la entrada dataframe usando la funcion "len" y la almacena como una variable de instancia "self.len"
        self.data = dataframe       # se asigna la entrada dataframe a una variable de instancia "self.data"
        
    def __getitem__(self, index):   # define el método "__getitem__" que toma un solo argumento index
        ''' el metodo __getitem__ devuelve un diccionario que contiene cuatro claves: 'input_ids', 'attention_mask', 'labels'y 'added_features' '''
        input_ids = torch.tensor(self.data.text_vec.iloc[index]).cpu() # almacena las características de los datos de "text_vec" ​​que se han convertido en un vector de longitud fija.
        attention_mask = torch.ones([input_ids.size(0)]).cpu()  # attention_mask almacena los elementos de entrada que se debe prestar atención y cuáles se deben ignorar
        label = self.data.same.iloc[index]              # almacena un valor escalar que representa la etiqueta de salida para la puntuación de complejidad
        targets = torch.tensor([1 - label, label])  #ojo probar ESTO ES NUEVO
        return {
            'input_ids': input_ids,               # devuelve las características de entrada para el punto de datos
            'attention_mask': attention_mask,     # devuelve la máscara de atención para el punto de datos
            'labels': targets                    # devuelve un valor escalar que representa la puntuación de complejidad
        }
            
    def __len__(self):    
        return self.len   # devuelve la longitud del conjunto de datos personalizado