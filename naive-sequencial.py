# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:37:11 2017

@author: dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

traindataframe = pd.read_csv('train-W.csv')
testdataframe = pd.read_csv('test-W.csv')

traindata = traindataframe[['Store', 'Dept', 'Date', 'Weekly_Sales', 'Week2']]
testdata = testdataframe[['Store', 'Dept', 'Date', 'Week']]
