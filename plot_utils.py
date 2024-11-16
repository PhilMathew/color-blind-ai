import os
from typing import *
import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
import seaborn as sns


def plot_history(
    train_hist: Dict[str, Sequence[float]], 
    save_path: Optional[str | os.PathLike] = 'train_hist.png'
) -> None:
    """
    Plots the training history of a model.

    :param train_hist: Dictionary of training history
    :type train_hist: Dict[str, Sequence[float]]
    :param save_path: Path to save the plot, defaults to 'train_hist.png'
    :type save_path: str or os.PathLike, optional
    """
    plot_val = 'val_loss' in train_hist.keys()
    
    if plot_val:
        fig, ((train_loss_ax, train_acc_ax), (val_loss_ax, val_acc_ax)) = plt.subplots(2, 2, figsize=(20, 10))
        
        val_loss_ax.plot(train_hist['val_loss'])
        val_loss_ax.set(title='Validation Loss', xlabel='Epoch', ylabel='Loss')
        
        val_acc_ax.plot(train_hist['val_acc'])
        val_acc_ax.set(title='Validation Accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=[0, 1])
    else:
        fig, (train_loss_ax, train_acc_ax) = plt.subplots(1, 2, figsize=(20, 10))
    
    train_loss_ax.plot(train_hist['train_loss'])
    train_loss_ax.set(title='Train Loss', xlabel='Epoch', ylabel='Loss')
    
    train_acc_ax.plot(train_hist['train_acc'])
    train_acc_ax.set(title='Train Accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=[0, 1])

    fig.savefig(str(save_path))


def plot_confmat(
    cm: Union[List, np.ndarray], 
    save_path: Optional[str | os.PathLike] = 'confmat.png', 
    title: Optional[str] = '', 
    label_mapping: Optional[Dict[Any, Any]] = None
) -> None:
    """
    Plots a given confusion matrix

    :param cm: Confusion matrix to plot
    :type cm: Union[List, np.ndarray]
    :param save_path: Path to save the plot, defaults to 'confmat.png'
    :type save_path: str or os.PathLike, optional
    :param title: Plot title, defaults to ''
    :type title: str, optional
    :param label_mapping: Dictionary mapping labels to their names, e.g. {0: 'cat', 1: 'dog'}, defaults to None
    :type label_mapping: dict, optional
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    cm = np.array(cm)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize the confusion matrix
    cm_norm[np.isnan(cm_norm)] = 0

    annot = np.zeros_like(cm, dtype=object)
    for i in range(annot.shape[0]):  # Creates an annotation array for the heatmap
        for j in range(annot.shape[1]):
            annot[i][j] = f'{cm[i][j]}\n{round(cm_norm[i][j] * 100, ndigits=3)}%'
    
    ax = sns.heatmap(cm_norm, annot=annot, fmt='', cbar=True, cmap=plt.cm.magma, vmin=0, vmax=1, ax=ax) # plot the confusion matrix
    ax.set(xlabel='Predicted Label', ylabel='Actual Label', title=f'{title} (CM Trace: {cm_norm.trace():.4f})')
    
    if label_mapping:
        ticks = sorted(label_mapping.keys(), key=(lambda x: label_mapping[x]))
        ax.set(xticklabels=ticks, yticklabels=ticks)
    
    # fig.tight_layout()
    fig.savefig(str(save_path))


def plot_roc_curve(
    fpr: ArrayLike, 
    tpr: ArrayLike, 
    save_path: Optional[str | os.PathLike] = 'roc_curve.png'
) -> None:
    """
    Plots a ROC curve

    :param fpr: False positive rate
    :type fpr: ArrayLike
    :param tpr: True positive rate
    :type tpr: ArrayLike
    :param save_path: Path to save output plot, defaults to 'roc_curve.png'
    :type save_path: str or os.PathLike, optional
    """
    fig, ax = plt.subplots(1, 1)
   
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')  # plot the baseline line
    ax.set(
        xlabel='False Positive Rate',
        xscale='log', 
        ylabel='True Positive Rate', 
        title=f'AUC = {np.trapz(tpr, fpr):.4f}'
    )
    fig.savefig(str(save_path))
