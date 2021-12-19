import os

file_date = "15_12_2021"
path_root = os.path.dirname(os.getcwd())
path_data = os.path.join(os.path.dirname(path_root), f"data/gnps_{file_date}/positive_mode/")



assert os.path.exists(os.path.join(path_data, f"GNPS_{file_date}.sqlite"))
assert os.path.exists(os.path.join(path_data, f"spec2vec_models/spec2vec_model_GNPS_{file_date}.model"))
assert os.path.exists(os.path.join(path_data, f"trained_models/ms2ds_GNPS_{file_date}.hdf5"))
assert os.path.exists(os.path.join(path_data, f"GNPS_{file_date}_ms2ds_embeddings.pickle"))
assert os.path.exists(os.path.join(path_data, f"GNPS_{file_date}_s2v_embeddings.pickle"))
assert os.path.exists(os.path.join(path_data, "ms2query_library_files/ms2q_train_spectra.pickle"))
assert os.path.exists(os.path.join(path_data, "ms2query_library_files/ms2q_val_spectra.pickle"))
assert os.path.exists(os.path.join(path_data, f"GNPS_{file_date}_pos_tanimoto_scores.pickle"))

from ms2query.select_data_for_training_ms2query_nn import DataCollectorForTraining

collector = DataCollectorForTraining(
    os.path.join(path_data, f"GNPS_{file_date}.sqlite"),
    os.path.join(path_data, f"spec2vec_models/spec2vec_model_GNPS_{file_date}.model"),
    os.path.join(path_data, f"trained_models/ms2ds_GNPS_{file_date}.hdf5"),
    os.path.join(path_data, f"GNPS_{file_date}_s2v_embeddings.pickle"),
    os.path.join(path_data, f"GNPS_{file_date}_ms2ds_embeddings.pickle"),
    os.path.join(path_data, "ms2query_library_files/ms2q_train_spectra.pickle"),
    os.path.join(path_data, "ms2query_library_files/ms2q_val_spectra.pickle"),
    os.path.join(path_data, f"GNPS_{file_date}_pos_tanimoto_scores.pickle"),
    preselection_cut_off=100)

collector.create_train_and_val_data(os.path.join(path_data, "ms2query_library_files/training_data.pickle"))