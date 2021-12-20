import os

file_date = "15_12_2021"

path_data = "C:\\HSD\\OneDrive - Hochschule Düsseldorf\\Data\\ms2query"



assert os.path.exists(os.path.join(path_data, "library_GNPS_15_12", f"library_GNPS_{file_date}.sqlite"))
assert os.path.exists(os.path.join(path_data, "trained_models", f"spec2vec_model_GNPS_{file_date}.model"))
assert os.path.exists(os.path.join(path_data, "trained_models", f"ms2ds_GNPS_{file_date}.hdf5"))
assert os.path.exists(os.path.join(path_data, "library_GNPS_15_12", f"library_GNPS_{file_date}_ms2ds_embeddings.pickle"))
assert os.path.exists(os.path.join(path_data, "library_GNPS_15_12", f"library_GNPS_{file_date}_s2v_embeddings.pickle"))
assert os.path.exists(os.path.join(path_data, "ms2q_train_spectra.pickle"))
assert os.path.exists(os.path.join(path_data, "ms2q_val_spectra.pickle"))
assert os.path.exists(os.path.join(path_data, f"GNPS_{file_date}_pos_tanimoto_scores.pickle"))

from ms2query.select_data_for_training_ms2query_nn import DataCollectorForTraining

collector = DataCollectorForTraining(
    os.path.join(path_data, "library_GNPS_15_12", f"library_GNPS_{file_date}.sqlite"),
    os.path.join(path_data, "trained_models", f"spec2vec_model_GNPS_{file_date}.model"),
    os.path.join(path_data, "trained_models", f"ms2ds_GNPS_{file_date}.hdf5"),
    os.path.join(path_data, "library_GNPS_15_12", f"library_GNPS_{file_date}_ms2ds_embeddings.pickle"),
    os.path.join(path_data, "library_GNPS_15_12", f"library_GNPS_{file_date}_s2v_embeddings.pickle"),
    os.path.join(path_data, "ms2q_train_spectra.pickle"),
    os.path.join(path_data, "ms2q_val_spectra.pickle"),
    os.path.join(path_data, f"GNPS_{file_date}_pos_tanimoto_scores.pickle"),
    preselection_cut_off=100)

collector.create_train_and_val_data(os.path.join(path_data,
                                                 "training_data.pickle"))