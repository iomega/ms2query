import os
import pickle
import pandas as pd
from typing import Union, List, Tuple, Dict
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from matplotlib import pyplot as plt


def create_ms2query_nn(layers: List[int],
                       input_dimensions: int,
                       model_loss: str = 'mean_squared_error',
                       activations: str = 'relu', last_activation=None
                       ) -> Sequential:
    """Creates a keras deep NN

        Args
        ----
        layers:
            The number of layers is the length of this list while the elements
            are the amount of neurons per layer.
        input_dimensions:
            The dimensions of the first layer
        model_loss:
            Loss function, default: "mean_squared_error"
        activations:
            The activation of the layers except the last one, default: "relu"
        last_activation:
            Activation of last layer, default: None
        """
    # define the keras model
    nn_model = Sequential()
    # add first layer
    nn_model.add(Dense(layers[0], input_dim=input_dimensions,
                       activation=activations))
    nn_model.add(Dropout(0.2))

    # add other layers
    for i in range(1, len(layers) - 1):  # skip first and last one
        nn_model.add(Dense(layers[i], activation=activations))
        nn_model.add(Dropout(0.2))
    # add last layer
    nn_model.add(Dense(layers[-1], activation=last_activation))
    # compile the keras model
    nn_model.compile(loss=model_loss, optimizer='adam',
                     metrics=['mae'])
    return nn_model


def train_ms2query_nn(nn_model: Sequential,
                      x_train, y_train, x_val, y_val,
                      model_epochs: int = 100, model_batch_size: int = 16,
                      save_name: Union[None, str] = None):
    """Train a keras deep NN, returns (model, history)

    Args
    ----
    nn_model:
        The model that should be trained
    x_train:
        The training set. Contains scores for matches between spectra.
    y_train:
        The training labels, containing the tanimoto scores corresponding to
        the matches in the training set.
    x_val:
        Validation set. Contains scores for matches between spectra.
    y_val:
        Validation labels, containing the tanimoto scores corresponding to the
        matches in the validation set.
    model_epochs:
        Number of epochs, default: 100
    model_batch_size:
        Batch size for updating the model, default: 16
    save_name:
        Location for saving model, result is not saved when None. Default=None
    """
    # pylint: disable=too-many-arguments
    earlystopper = EarlyStopping(monitor='val_loss', patience=10,
                                 verbose=1)
    if save_name and not os.path.exists(save_name + ".hdf5"):
        checkpointer = ModelCheckpoint(filepath=save_name + ".hdf5",
                                       monitor='val_loss', verbose=1,
                                       save_best_only=True)
        callbacks = [earlystopper, checkpointer]
    else:
        callbacks = earlystopper
    # fit the keras model on the dataset
    hist = nn_model.fit(x_train, y_train, epochs=model_epochs,
                        batch_size=model_batch_size,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks)
    history = hist.history

    if save_name and not os.path.exists(save_name + '_train_hist.pickle'):
        with open(save_name + '_train_hist.pickle', 'wb') as hist_outf:
            pickle.dump(history, hist_outf)
    return nn_model, history


def create_and_train_ms2query_nn(x_train: pd.DataFrame, y_train: pd.DataFrame,
                                 x_val: pd.DataFrame, y_val: pd.DataFrame,
                                 layers: List[int],
                                 model_loss: str = 'mean_squared_error',
                                 activations: str = 'relu',
                                 last_activation=None,
                                 model_epochs: int = 100,
                                 model_batch_size: int = 16,
                                 save_name: Union[None, str] = None
                                 ) -> Tuple[Sequential, Dict[str, float]]:
    """Create and tain a keras deep NN, returns (model, history)

    Args
    ----
    x_train:
        The training set. Contains scores for matches between spectra.
    y_train:
        The training labels, containing the tanimoto scores corresponding to
        the matches in the training set.
    x_val:
        Validation set. Contains scores for matches between spectra.
    y_val:
        Validation labels, containing the tanimoto scores corresponding to the
        matches in the validation set.
    layers:
        The number of layers is the length of this list while the elements
        are the amount of neurons per layer.
    model_loss:
        Loss function, default: "mean_squared_error"
    activations:
        The activation of the layers except the last one, default: "relu"
    last_activation:
        Activation of last layer, default: None
    model_epochs:
        Number of epochs, default: 100
    model_batch_size:
        Batch size for updating the model, default: 16
    save_name:
        Location for saving model, result is not saved when None. Default=None
    """
    # pylint: disable=too-many-arguments
    nn_model = create_ms2query_nn(layers, x_train.shape[1], model_loss,
                                  activations, last_activation)
    nn_model, history = train_ms2query_nn(nn_model, x_train, y_train, x_val,
                                          y_val, model_epochs,
                                          model_batch_size, save_name)
    return nn_model, history


def plot_history(history):
    """Plots the MAE and MSE loss for the history of a trained model"""
    _, (ax1, ax2) = plt.subplots(2, 1, sharex="all", figsize=(12, 8),
                                   dpi=100)

    ax1.plot(history['mae'], "o--", label='Acuracy (training data)')
    ax1.plot(history['val_mae'], "o--", label='Acuracy (validation data)')
    ax1.set_title('MAE loss')
    ax1.set_ylabel("MAE")
    ax1.legend()

    ax2.plot(history['loss'], "o--", label='training data')
    ax2.plot(history['val_loss'], "o--", label='validation data')
    ax2.set_title('MSE loss')
    ax2.set_ylabel("MSE loss")
    ax2.set_xlabel("epochs")
    ax2.legend()
    plt.show()
