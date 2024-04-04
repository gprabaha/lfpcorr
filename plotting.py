#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:46:30 2024

@author: pg496
"""

import matplotlib.pyplot as plt

def plot_training_validation_losses(history, title, ax):
    """
    Plots training and validation losses.

    Args:
        history (keras.callbacks.History): History object returned by the fit method.
        title (str): Title indicating the model combination.
        ax (matplotlib.axes.Axes): Axes object to plot on.
    """
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)