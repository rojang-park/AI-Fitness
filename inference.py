from socket import INADDR_ALLHOSTS_GROUP
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import datetime
from train import trainer

class inference():
    def __init__(self):
        self.input_path = '../'
        self.model_path = '../save/'
        self.output_path = '../save/'
        

    def predict(self):
        model = torch.load(self.model_path + 'model.pt')
        self.input2 = pd.read_csv(self.input_path + '/input.csv', names=[1,2,3,4], encoding='utf-8-sig')
        input_data = np.array(self.input2)
        input_data = torch.from_numpy(input_data).float()
        pred = model.predict(input_data)
        self.outlier = np.argmax(pred.detach().numpy(), axis=1)


    def run(self):
        self.predict()

if __name__ == "__main__":
    my_tester = inference()
    my_tester.run()
