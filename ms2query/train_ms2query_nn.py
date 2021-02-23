from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense

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