# import os
# from ms2query.create_new_library.train_ms2query_model import \
#     DataCollectorForTraining
#
#
# file_date = "15_12_2021"
# path_root = os.path.dirname(os.getcwd())
# path_data = os.path.join(path_root, "../../../data/libraries_and_models/gnps_15_12_2021/in_between_files")
# path_negative_mode = os.path.join(path_data, "../negative_mode_models")
#
# assert os.path.exists(os.path.join(path_negative_mode, f"neg_library_GNPS_{file_date}.sqlite"))
# assert os.path.exists(os.path.join(path_negative_mode, f"neg_mode_spec2vec_model_GNPS_{file_date}.model"))
# assert os.path.exists(os.path.join(path_negative_mode, f"ms2ds_GNPS_{file_date}.hdf5"))
# assert os.path.exists(os.path.join(path_negative_mode, f"neg_library_GNPS_{file_date}_s2v_embeddings.pickle"))
# assert os.path.exists(os.path.join(path_negative_mode, f"neg_library_GNPS_{file_date}_ms2ds_embeddings.pickle"))
# assert os.path.exists(os.path.join(path_negative_mode, "neg_ms2q_train_spectra.pickle"))
# assert os.path.exists(os.path.join(path_negative_mode, "neg_ms2q_val_spectra.pickle"))
# assert os.path.exists(os.path.join(path_data, f"GNPS_{file_date}_neg_tanimoto_scores.pickle"))
#
# collector = DataCollectorForTraining(os.path.join(path_negative_mode, f"neg_library_GNPS_{file_date}.sqlite"),
#                                      os.path.join(path_negative_mode,
#                                                   f"neg_mode_spec2vec_model_GNPS_{file_date}.model"),
#                                      os.path.join(path_negative_mode, f"ms2ds_GNPS_{file_date}.hdf5"),
#                                      os.path.join(path_negative_mode,
#                                                   f"neg_library_GNPS_{file_date}_s2v_embeddings.pickle"),
#                                      os.path.join(path_negative_mode,
#                                                   f"neg_library_GNPS_{file_date}_ms2ds_embeddings.pickle"),
#                                      os.path.join(path_negative_mode, "neg_ms2q_train_spectra.pickle"),
#                                      os.path.join(path_negative_mode, "neg_ms2q_val_spectra.pickle"),
#                                      os.path.join(path_data, f"GNPS_{file_date}_neg_tanimoto_scores.pickle"),
#                                      preselection_cut_off=100)
#
# collector.create_train_and_val_data(os.path.join(path_data,
#                                                  "neg_final_ms2q_training_data.pickle"))
