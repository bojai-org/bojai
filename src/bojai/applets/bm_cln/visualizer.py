import os
import matplotlib.pyplot as plt
from trainer import Logger
from statistics import mean
from global_vars import browseDict
class Visualizer:
    '''
    This class visualizes different models. It uses matplotlib.

    The class needs a Logger which should be defined in the trainer.py file. This class provides different ways of visualising the logged data. 

    It will support visualization to the following: 
     - change of loss by epoch 
     - validation vs training loss per epoch 
     - evaluation data vs training data 
     - evaluation data vs validation data
    '''
    def __init__(self, save_dir=None):
        self.logger : Logger = Logger()
        self.save_dir = save_dir or "./visuals"
        os.makedirs(self.save_dir, exist_ok=True)
        print("AAAAAAAAAAAAAAAAAAAAAa")

    '''
    plots the loss vs epochs 
    '''
    def plot_loss(self):
        fig, ax = plt.subplots()
        epochs = sorted(self.logger.get_logs().keys())
        if -1 in epochs:
            epochs.remove(-1)
        print(epochs)
        print(self.logger.get_logs()[0])
        print(self.logger.get_logs()[0])
        train_losses = [self.logger.get_logs()[ep].get('loss') for ep in epochs]
        ax.plot(epochs, train_losses, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Epoch')
        ax.legend()
        plt.show()

    '''
    plots validation and training error vs epochs
    '''
    def plot_validation_vs_training(self):
        matrice = browseDict['eval matrice']
        fig, ax = plt.subplots()
        epochs = sorted(self.logger.get_logs().keys())
        if -1 in epochs:
            epochs.remove(-1)
        print(self.logger.get_logs()[0])
        print(self.logger.get_logs()[0])
        train = [self.logger.get_logs()[ep].get('train') for ep in epochs]
        valid = [self.logger.get_logs()[ep].get('valid') for ep in epochs]
        ax.plot(epochs, train, label='Training Loss')
        ax.plot(epochs, valid, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(matrice)
        ax.set_title('Training vs Validation Loss')
        ax.legend()
        plt.show()

    def plot_train_vs_eval(self, metric_name = browseDict['eval matrice']):
        '''
        Plots a bar chart comparing the mean training score vs evaluation score
        '''
        fig, ax = plt.subplots()
        logs = self.logger.get_logs()

        epochs = sorted(logs.keys())
        if -1 in epochs:
            epochs.remove(-1)
        train_scores = [logs[ep].get('train', 0) for ep in epochs]
        mean_train_score = mean(train_scores)

        # Get the latest evaluation score from the last epoch
        eval_score = self.logger.get_logs()[-1]

        # Bar labels and values
        labels = ['Mean Train', 'Eval']
        scores = [mean_train_score, eval_score]

        ax.bar(labels, scores)
        ax.set_ylabel(metric_name)
        ax.set_title(f'Mean Train score vs Eval')
        plt.tight_layout()
        plt.show()

    '''
    plots evaluation data vs validation
    '''
    def plot_valid_vs_eval(self, metric_name = browseDict['eval matrice']):
        '''
        Plots a bar chart comparing the mean validation score vs evaluation score
        '''
        fig, ax = plt.subplots()
        logs = self.logger.get_logs()

        epochs = sorted(logs.keys())
        if -1 in epochs:
            epochs.remove(-1)
        train_scores = [logs[ep].get('valid', 0) for ep in epochs]
        mean_train_score = mean(train_scores)

        # Get the latest evaluation score from the last epoch
        eval_score = self.logger.get_logs()[-1]

        # Bar labels and values
        labels = ['Mean Valid', 'Eval']
        scores = [mean_train_score, eval_score]

        ax.bar(labels, scores)
        ax.set_ylabel(metric_name)
        ax.set_title(f'Mean Valid score vs Eval')
        plt.tight_layout()
        plt.show()

    '''
    saves the plot in the directory.
    '''
    def save_plot(self, fig, filename):
        fig_path = os.path.join(self.save_dir, filename)
        fig.savefig(fig_path)