from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense
from ms2query.app_helpers import load_pickled_file
import pandas as pd


def train_nn(x_train, y_train, x_test, y_test,
             layers=[12, 12, 12, 12, 12, 1],
             model_loss='binary_crossentropy', activations='relu',
             last_activation='sigmoid', model_epochs=20,
             model_batch_size=16,
             save_name=False):
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
                     metrics=['accuracy'])
    # fit the keras model on the dataset
    hist = nn_model.fit(x_train, y_train, epochs=model_epochs,
                        batch_size=model_batch_size)
    history = hist.history

    # training set
    print('Training loss: {:.4f}\n'.format(history['loss'][-1]))

    # test set
    loss, accuracy = nn_model.evaluate(x_test, y_test)
    print('Test accuracy: {:.2f}'.format(accuracy * 100))
    print('Test loss: {:.4f}'.format(loss))

    # if save_name and not os.path.exists(save_name):
    #     print('Saving model at:', save_name)
    #     nn_model.save(save_name)
    #     with open(save_name + '_train_hist.pickle', 'wb') as hist_outf:
    #         pickle.dump(history, hist_outf)

    return nn_model, history, accuracy, loss

def run_test_file():
    training_data = "../downloads/models/spec2vec_models/train_nn_model_data/nn_prep_training_found_matches_s2v_2dec.pickle"
    testing_data = "../downloads/models/spec2vec_models/train_nn_model_data/nn_prep_testing_found_matches_s2v_2dec.pickle"

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
    train_nn(x_train[:20], y_train[:20], x_test, y_test)


if __name__ == "__main__":
    file_name = "test_matches_info_training_and_testing.pickle"
    training_set, training_labels, testing_set, testing_labels = \
        load_pickled_file(file_name)

    train_nn(training_set,
             training_labels,
             testing_set,
             testing_labels)
    # run_test_file()