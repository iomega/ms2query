"""
Choices made, different than joris method:
The test data set used bij Joris named nn_prep_training_found_matches_s2v_2dec.pickle
contains 2 sets of 2000 spectra the second set is testing data. This set
consists of 1000 spectra that have more than 4 inchikeys and 1000 spectra that
have only 1 inchikey. From this set a validation set is selected of 500 spectra.
To make sure this set is similar, 250 were taken from the first 1000 and 250
were taken from the second 1000. In the functions here this is not, the case,
since later sets will probably just have random spectra.

"""
import pickle
import os
from typing import List
import numpy as np
import pandas as pd
from matchms.typing import SpectrumType
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense
from ms2query.app_helpers import load_pickled_file
from ms2query.ms2library import Ms2Library, get_spectra_from_sqlite
from ms2query.query_from_sqlite_database import \
    get_tanimoto_score_for_inchikeys


class TrainImproveLibraryMatchingNN(Ms2Library):
    def __init__(self,
                 sqlite_file_location: str,
                 s2v_model_file_name: str,
                 ms2ds_model_file_name: str,
                 pickled_s2v_embeddings_file_name: str,
                 pickled_ms2ds_embeddings_file_name: str,
                 training_spectra_file_name: str,
                 **settings):
        self.training_spectra, test_and_val_spectra = \
            load_pickled_file(training_spectra_file_name)

        # Select random spectra for validation set.
        random_val_spectra_indexes = list(
            np.random.choice(range(0, len(test_and_val_spectra)),
                             500,
                             replace=False))
        self.test_spectra = [spectrum
                             for i, spectrum in enumerate(test_and_val_spectra)
                             if i not in random_val_spectra_indexes]
        self.validation_spectra = [spectrum for i, spectrum
                                   in enumerate(test_and_val_spectra)
                                   if i in random_val_spectra_indexes]

        self.training_spectra = self.training_spectra[:20]
        self.test_spectra = self.test_spectra[:15]
        self.validation_spectra = self.validation_spectra[:5]
        print(type(self.validation_spectra[0]))
        print("ingeladen")

        super().__init__(sqlite_file_location,
                         s2v_model_file_name,
                         ms2ds_model_file_name,
                         pickled_s2v_embeddings_file_name,
                         pickled_ms2ds_embeddings_file_name,
                         **settings)

    def train_network(self):
        self.get_spectra_info_for_training(self.test_spectra)

    def get_spectra_info_for_training(self,
                                      query_spectra: List[SpectrumType]):

        # Does a preselection of the 20 spectra with the highest ms2ds score
        # and add info.
        query_spectra_matches_info = \
            self.collect_matches_data_multiple_spectra(query_spectra)

        query_spectra_ids = [spectrum.get("spectrum_id") for spectrum
                             in query_spectra]
        tanimoto_scores = pd.Dataframe()
        for query_spectrum_id in query_spectra_ids:
            match_spectrum_ids = query_spectra_matches_info[query_spectrum_id]["spectrum_id"]
            tanimoto_scores.append(
                get_tanimoto_score_for_inchikeys(query_spectra_ids,
                                                 match_spectrum_ids,
                                                 self.sqlite_file_location))

        print(query_spectra_matches_info)
        print(tanimoto_scores)

    def train_nn(self, X_train, y_train, X_test, y_test,
                 layers=[12, 12, 12, 12, 12, 1],
                 model_loss='binary_crossentropy', activations='relu',
                 last_activation='sigmoid', model_epochs=20,
                 model_batch_size=16,
                 save_name=False):
        """Train a keras deep NN and test on test data, returns (model, history, accuracy, loss)

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
        """

        # define the keras model
        nn_model = Sequential()
        # add first layer
        nn_model.add(Dense(layers[0], input_dim=X_train.shape[1],
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
        hist = nn_model.fit(X_train, y_train, epochs=model_epochs,
                            batch_size=model_batch_size)
        history = hist.history

        # training set
        print('Training loss: {:.4f}\n'.format(history['loss'][-1]))

        # test set
        loss, accuracy = nn_model.evaluate(X_test, y_test)
        print('Test accuracy: {:.2f}'.format(accuracy * 100))
        print('Test loss: {:.4f}'.format(loss))

        if save_name and not os.path.exists(save_name):
            print('Saving model at:', save_name)
            nn_model.save(save_name)
            with open(save_name + '_train_hist.pickle', 'wb') as hist_outf:
                pickle.dump(history, hist_outf)

        return nn_model, history, accuracy, loss

if __name__ == "__main__":
    sqlite_file_name = \
        "../downloads/data_all_inchikeys_with_tanimoto_and_parent_mass.sqlite"
    s2v_model_file_name = \
        "../downloads/" \
        "spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"
    s2v_pickled_embeddings_file = \
        "../downloads/embeddings_all_spectra.pickle"
    ms2ds_model_file_name = \
        "../../ms2deepscore/data/" \
        "ms2ds_siamese_210207_ALL_GNPS_positive_L1L2.hdf5"
    ms2ds_embeddings_file_name = \
        "../../ms2deepscore/data/ms2ds_embeddings_2_spectra.pickle"
    neural_network_model_file_location = \
        "../model/nn_2000_queries_trimming_simple_10.hdf5"
    training_spectra_file_name = \
        "../downloads/models/spec2vec_models/train_nn_model_data/test_and_validation_spectra_nn_model.pickle"
    # Create library object
    my_library = TrainImproveLibraryMatchingNN(
        sqlite_file_name,
        s2v_model_file_name,
        ms2ds_model_file_name,
        s2v_pickled_embeddings_file,
        ms2ds_embeddings_file_name,
        training_spectra_file_name)

    print(my_library.train_network())
