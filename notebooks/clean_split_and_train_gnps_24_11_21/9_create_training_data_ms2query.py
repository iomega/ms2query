import os
path_root = os.path.dirname(os.getcwd())
path_data = os.path.join(os.path.dirname(path_root), "data/gnps_24_11_2021/positive_mode/")

assert os.path.exists(os.path.join(path_data, "GNPS_24_11_2021.sqlite"))
assert os.path.exists(os.path.join(path_data, "spec2vec_models/spec2vec_model_GNPS_24_11_2021.model"))
assert os.path.exists(os.path.join(path_data, "trained_models/ms2ds_GNPS_25_11_2021.hdf5"))
assert os.path.exists(os.path.join(path_data, "GNPS_24_11_2021_ms2ds_embeddings.pickle"))
assert os.path.exists(os.path.join(path_data, "GNPS_24_11_2021_s2v_embeddings.pickle"))
assert os.path.exists(os.path.join(path_data, "ms2query_library_files/ms2q_train_spectra.pickle"))
assert os.path.exists(os.path.join(path_data, "ms2query_library_files/ms2q_val_spectra.pickle"))
assert os.path.exists(os.path.join(path_data, "GNPS_24_11_2021_pos_tanimoto_scores.pickle"))

from ms2query.select_data_for_training_ms2query_nn import DataCollectorForTraining

collector = DataCollectorForTraining(
    os.path.join(path_data, "GNPS_24_11_2021.sqlite"),
    os.path.join(path_data, "spec2vec_models/spec2vec_model_GNPS_24_11_2021.model"),
    os.path.join(path_data, "trained_models/ms2ds_GNPS_25_11_2021.hdf5"),
    os.path.join(path_data, "GNPS_24_11_2021_s2v_embeddings.pickle"),
    os.path.join(path_data, "GNPS_24_11_2021_ms2ds_embeddings.pickle"),
    os.path.join(path_data, "ms2query_library_files/ms2q_train_spectra.pickle"),
    os.path.join(path_data, "ms2query_library_files/ms2q_val_spectra.pickle"),
    os.path.join(path_data, "GNPS_24_11_2021_pos_tanimoto_scores.pickle"),
    preselection_cut_off=100)

collector.create_train_and_val_data(os.path.join(path_data, "ms2query_library_files/training_data.pickle"))