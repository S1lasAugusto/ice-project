# This will throw import warning, but worry not, this file will be moved into the PonyGE2 folder afterwards

import pandas as pd
from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff

from .model_factory import ModelFactory

class scorer_fitness(base_ff):

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.load_data()

    def load_data(self):

        self.train_X = pd.read_csv(params["DATASET_TRAIN"], sep='\t', encoding='ISO-8859-1')

        self.train_Y = self.train_X['domain1_score']
        self.train_X = self.train_X.dropna(axis=1)
        self.train_X = self.train_X.drop(columns=['rater1_domain1', 'rater2_domain1'])

    def evaluate(self, ind):
        
        data = self.train_X
        labels = self.train_Y

        embedding, network = ind.phenotype.split(':')

        model = ModelFactory(embedding_type=embedding, model_type=network, dataset=data, target=labels)

        _, mse = model.train(params['EMBEDDINGS_PATH'])

        return mse
