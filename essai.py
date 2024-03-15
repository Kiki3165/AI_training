from utilities import *
import numpy as np
import matplotlib as plt
from connexions import artificial_neuron

X_train, y_train, X_test, y_test = load_data()

W, b = artificial_neuron(X_train, y_train)