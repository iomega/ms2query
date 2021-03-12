import os
import pickle
from typing import Union
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from ms2query.app_helpers import load_pickled_file


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,8), dpi=100)

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


def train_nn(x_train, y_train, x_val, y_val,
             layers=[48, 48, 1],
             model_loss='mean_squared_error', activations='relu',
             last_activation=None, model_epochs=100,
             model_batch_size=16,
             save_name: Union[None, str] = None):
    '''Train a keras deep NN and test on test data, returns (model, history, accuracy, loss)

    X_train: matrix like object like pd.DataFrame, training set
    y_train: list like object like np.array, training labels
    X_test: matrix like object like pd.DataFrame, test set
    y_test: list like object like np.array, test labels
    layers: list of ints, the number of layers is the len of this list while the elements
        are the amount of neurons per layer, default: [12, 12, 12, 12, 12, 1]
    model_loss: str, loss function, default: binary_crossentropy
    activations: str, the activation of the layers except the last one, default: relu
    last_activation: str, activation of last layer, default: sigmoid
    model_epochs: int, number of epochs, default: 20
    model_batch_size: int, batch size for updating the model, default: 16
    save_name: str, location for saving model, optional, default: False

    Returns:
    model: keras sequential
    history: dict, training statistics
    accuracy: float, accuracy on test set
    loss, float, loss on test set

    If save_name is not False and save_name exists this function will load existing model
    '''

    # define the keras model
    nn_model = Sequential()
    # add first layer
    nn_model.add(Dense(layers[0], input_dim=x_train.shape[1],
                       activation=activations))
    # add other layers
    for i in range(1, len(layers) - 1):  # skip first and last one
        nn_model.add(Dense(layers[i], activation=activations))
    # add last layer
    nn_model.add(Dense(layers[-1], activation=last_activation))
    # compile the keras model
    nn_model.compile(loss=model_loss, optimizer='adam',
                     metrics=['mae'])


    earlystopper = EarlyStopping(monitor='val_loss', patience=10,
                                 verbose=1)  # patience - try x more epochs to improve val_loss
    checkpointer = ModelCheckpoint(filepath=save_name + ".hdf5",
                                   monitor='val_loss', verbose=1,
                                   save_best_only=True)

    # fit the keras model on the dataset
    hist = nn_model.fit(x_train, y_train, epochs=model_epochs,
                        batch_size=model_batch_size,
                        validation_data=(x_val, y_val),
                        callbacks=[earlystopper, checkpointer])
    history = hist.history

    if save_name and not os.path.exists(save_name):
        with open(save_name + '_train_hist.pickle', 'wb') as hist_outf:
            pickle.dump(history, hist_outf)
    return nn_model, history

def run_test_file():
    training_data = "../downloads/models/train_nn_model_data/nn_prep_training_found_matches_s2v_2dec.pickle"
    testing_data = "../downloads/models/train_nn_model_data/nn_prep_testing_found_matches_s2v_2dec.pickle"

    training_set = load_pickled_file(training_data)
    testing_set = load_pickled_file(testing_data)
    nn_training_found_matches_s2v_2dec = training_set[0].append(
        training_set[1:])
    nn_training_found_matches_s2v_2dec = \
        nn_training_found_matches_s2v_2dec.sample(frac=1)

    nn_testing_full_found_matches_s2v_2dec = \
        testing_set[0].append(
            testing_set[1:])
    nn_testing_full_found_matches_s2v_2dec = nn_testing_full_found_matches_s2v_2dec.sample(
        frac=1)

    x_train = nn_training_found_matches_s2v_2dec.drop(['similarity', 'label'],
                                                      axis=1)
    y_train = nn_training_found_matches_s2v_2dec['similarity']
    x_test = nn_testing_full_found_matches_s2v_2dec.drop(
        ['similarity', 'label'],
        axis=1)
    y_test = nn_testing_full_found_matches_s2v_2dec['similarity']
    print(nn_training_found_matches_s2v_2dec)
    print(type(nn_training_found_matches_s2v_2dec))
    print(nn_testing_full_found_matches_s2v_2dec)
    train_nn(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    # file_name = "../downloads/train_ms2query_nn_data/training_and_validations_sets.pickle"
    # training_set, training_labels, testing_set, testing_labels = \
    #     load_pickled_file(file_name)
    # import pandas as pd
    # pd.set_option("display.max_columns", 10)
    # print(training_set,
    #          training_labels,
    #          testing_set,
    #          testing_labels)
    # nn_model, history = train_nn(training_set,
    #                             training_labels,
    #                             testing_set,
    #                             testing_labels,
    #                             save_name="../downloads/train_ms2query_nn_data/test_models/ms2query_model")
    history_file = "../downloads/train_ms2query_nn_data/test_models/ms2query_model_train_hist.pickle"
    # history_file = "../downloads/train_ms2query_nn_data/test_models/test_mse_train_hist.pickle"
    history = load_pickled_file(history_file)
    plot_history(history)
