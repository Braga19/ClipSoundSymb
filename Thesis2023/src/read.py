import pandas as pd
import os
from google.colab import drive

def read_df(file_path):
    '''read a csv file and return a copy as a pandas dataset '''

    data = pd.read_csv(file_path)
    df = data.copy()

    return df

def mount():
  return drive.mount('/content/drive')