import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from tqdm import tqdm
from tkinter import *
from tkinter import filedialog

# Define a simple neural network for demonstration
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def preprocess_data(df):
  
    target_column = 'time'
    
  
    scaler = StandardScaler()
    features = df.drop(columns=[target_column])
    features = scaler.fit_transform(features)
    
    
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    
 
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
  
    X = torch.tensor(df_upsampled.drop(columns=[target_column]).values, dtype=torch.float32)
    y = torch.tensor(df_upsampled[target_column].values, dtype=torch.float32).view(-1, 1)
    
    return X, y

def train_model(X_train, y_train, epochs=10, lr=0.001):
    model = SimpleNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    return model

def plot_roc_curve(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def main():
   
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    df = pd.read_csv(file_path)

  
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = train_model(X_train, y_train)


    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    
  
    y_pred_np = y_pred.numpy()

    
    plot_roc_curve(y_test.numpy(), y_pred_np)

if __name__ == "__main__":
    main()
