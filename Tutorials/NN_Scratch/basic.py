import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

#Generating the dataset
def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def visualize(X, y, clf):
    plot_decision_boundary(lambda x:clf.predict(x), X, y)
    plt.title("Logistic Regression")

def plot_decision_boundary(pred_func, X, y):
    
