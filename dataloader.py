import torch
from torch.utils.data import Dataset, DataLoader
import os
import zipfile
import numpy as np
from scipy.optimize import curve_fit


num_cases = 1000
file_dir='/homes/lwjiang/Data/IVIM/public_training_data/'
fname_gt ='_IVIMParam.npy'
fname_tissue ='_TissueType.npy'
fname_noisyDWIk = '_NoisyDWIk.npy'
fname_gtimg = '_gtDWIs.npy'

def read_data(file_dir, fname, i):   
    fname_tmp = file_dir + "{:04}".format(i) + fname
    data = np.load(fname_tmp)  
    return data

for i in range(num_cases):        
        x = read_data(file_dir, fname_gt, i+1)
        k= read_data(file_dir, fname_noisyDWIk, i+1)
        arr3D_fittedParams = fit_biExponential_model(k, b)

class MyDataset(Dataset):
    def __init__(self, file_dir, fname_noisyDWIk, start_index, end_index):
        self.file_dir = file_dir
        self.fname_noisyDWIk = fname_noisyDWIk
        self.start_index = start_index
        self.end_index = end_index

    def __len__(self):
        return self.end_index - self.start_index + 1

    def __getitem__(self, idx):
        i = idx + self.start_index
        x = read_data(self.file_dir, self.fname_noisyDWIk, i)
        # Assuming you have a function to perform Fourier transform on x
        x_transformed = perform_fourier_transform(x)
        return x_transformed

# Assuming you have a function fit_biExponential_model defined
def fit_biExponential_model(data, b):
    # Your implementation here
    pass

# Set the number of cases for training and validation
num_train_cases = 800
num_val_cases = 200

# Create training dataset and dataloader
training_dataset = MyDataset(file_dir, fname_noisyDWIk, 1, num_train_cases)
training_dataloader = DataLoader(training_dataset, batch_size=4, shuffle=True)

# Create validation dataset and dataloader
validation_dataset = MyDataset(file_dir, fname_noisyDWIk, num_train_cases + 1, num_train_cases + num_val_cases)
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False)

# Now you can iterate over training_dataloader and validation_dataloader in your training loop
for epoch in range(num_epochs):
    for batch in training_dataloader:
        # Your training code here

    # Validation
    with torch.no_grad():
        for batch in validation_dataloader:
            # Your validation code here