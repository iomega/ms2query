# import os
# from ms2query.create_new_library.train_ms2query_model import \
#     DataCollectorForTraining
#
#
# file_date = "15_12_2021"
#
# path_data = "C:/Users/jonge094/PycharmProjects/PhD_MS2Query/ms2query/data/libraries_and_models/gnps_15_12_2021/"
# path_library = os.path.join(path_data, "library_GNPS_15_12")
# path_in_between = os.path.join(path_data, "in_between_files")
#
# assert os.path.exists(os.path.join(path_library, f"library_GNPS_{file_date}.sqlite"))
# assert os.path.exists(os.path.join(path_library, f"spec2vec_model_GNPS_{file_date}.model"))
# assert os.path.exists(os.path.join(path_library, f"ms2ds_model_GNPS_{file_date}.hdf5"))
# assert os.path.exists(os.path.join(path_library, f"library_GNPS_{file_date}_s2v_embeddings.pickle"))
# assert os.path.exists(os.path.join(path_library, f"library_GNPS_{file_date}_ms2ds_embeddings.pickle"))
# assert os.path.exists(os.path.join(path_in_between, "ms2q_train_spectra.pickle"))
# assert os.path.exists(os.path.join(path_in_between, "ms2q_val_spectra.pickle"))
# assert os.path.exists(os.path.join(path_in_between, f"GNPS_{file_date}_pos_tanimoto_scores.pickle"))
#
# collector = DataCollectorForTraining(os.path.join(path_data, "library_GNPS_15_12", f"library_GNPS_{file_date}.sqlite"),
#                                      os.path.join(path_library, f"spec2vec_model_GNPS_{file_date}.model"),
#                                      os.path.join(path_library, f"ms2ds_model_GNPS_{file_date}.hdf5"),
#                                      os.path.join(path_library, f"library_GNPS_{file_date}_s2v_embeddings.pickle"),
#                                      os.path.join(path_library, f"library_GNPS_{file_date}_ms2ds_embeddings.pickle"),
#                                      os.path.join(path_in_between, "ms2q_train_spectra.pickle"),
#                                      os.path.join(path_in_between, "ms2q_val_spectra.pickle"),
#                                      os.path.join(path_in_between, f"GNPS_{file_date}_pos_tanimoto_scores.pickle"),
#                                      preselection_cut_off=100)
#
# collector.create_train_and_val_data(os.path.join(path_data,
#                                                  "final_ms2q_training_data.pickle"))
