""" Class to create a StackedCLSModel object.
    This class is used to: 
    - create a StackedCLSModel object.
    - vectorize the texts.
    - train the model.
    - test the model.
"""

import re
from sklearn import datasets
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer


class StackedCLSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None
        self.tokenizer = None

        self.model_type = os.environ.get('MODEL_TYPE')

        if self.model_type == 'bert':
            self.bert()
        elif self.model_type == 'deberta':
            self.deberta()

    def vectorize_text(self, sequence1: str, sequence2: str, max_length):
        # Unicode normalization
        # s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')  # elimina cualquier diacrítico o acento de la cadena s

        # reemplaza todas las coincidencias del patrón con un espacio
        sequence1 = re.sub(r"[^a-zA-Záéíóú.,!?;:<>()$€\[\]]+", r" ", sequence1)
        sequence2 = re.sub(r"[^a-zA-Záéíóú.,!?;:<>()$€\[\]]+", r" ", sequence2)

        '''convierte la entrada de texto sin formato en un formato numérico que se puede introducir en un modelo de aprendizaje automático'''
        indexed_tokens = self.tokenizer.encode(  # utiliza el tokenizador previamente entrenado
            # para codificar una cadena "s" en sus identificadores de token correspondientes
            sequence1, sequence2,
            # especifica si se agregan tokens especiales al principio y al final de la secuencia de tokens
            add_special_tokens=True,
            # especifica la longitud máxima de la secuencia de tokens resultante
            max_length=max_length,
            # especifica cómo rellenar secuencias más cortas a la misma longitud que la secuencia más larga.
            padding='longest',
            # especifica si se truncan las secuencias que son más largas que "max_length"
            truncation=True,
            # especifica que la salida debe devolverse como una matriz numpy (o pt?).
            return_tensors='np'
        )
        return indexed_tokens[0]

    def bert(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.config = BertConfig.from_pretrained(
            'bert-base-uncased', output_hidden_states=True, output_attentions=True)
        self.MODEL = BertModel.from_pretrained(
            'bert-base-uncased', config=self.config)

    def berta(self):
        self.tokenizer = DebertaTokenizer.from_pretrained(
            "microsoft/deberta-base")
        self.config = DebertaConfig.from_pretrained(
            "microsoft/deberta-base", output_hidden_states=True, output_attentions=True)
        self.MODEL = DebertaModel.from_pretrained(
            "microsoft/deberta-base", config=self.config)
