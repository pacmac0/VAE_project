import json
import numpy as np
import matplotlib.pyplot as plt

def read_data_from_file(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

def plot_train(path):
    json = read_data_from_file(path)
    plt.title("Training Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.plot(json['train_loss'], label='Train-Loss')
    plt.plot(json['train_re'], label='Train-RE')
    plt.plot(json['train_kl'], label='Train-KL')
    plt.legend(('Train-Loss', 'Train-RE', 'Train-KL'))
    plt.savefig('plots/train_plot.png')
    plt.show()

def plot_test(path):
    json = read_data_from_file(path)
    plt.title("Testing Batches")
    plt.xlabel("Batches")
    plt.ylabel("Values")
    plt.plot(json['test_loss'], label='Test-Loss')
    plt.plot(json['test_re'], label='Test-RE')
    plt.plot(json['test_kl'], label='Test-KL')
    plt.legend(('Test-Loss', 'Test-RE', 'Test-KL'))
    plt.savefig('plots/test_plot.png')
    plt.show()

if __name__ == '__main__':
    plot_train(path='plots/lossvalues_train.json')
    plot_test(path='plots/lossvalues_test.json')