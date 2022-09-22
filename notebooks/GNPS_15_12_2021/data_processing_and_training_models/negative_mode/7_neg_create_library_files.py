# Deprecated use version 0.3.3 or older

# import os
# from ms2query.library_files_creator import LibraryFilesCreator
#
#
# file_date = "15_12_2021"
# path_root = os.path.dirname(os.getcwd())
# path_data = os.path.join(path_root, "../../../data/libraries_and_models/gnps_15_12_2021/in_between_files")
# path_library = os.path.join(path_data, "../negative_mode_models")
#
# file_creator = LibraryFilesCreator(os.path.join(path_data,
#                                                 f"GNPS_{file_date}_neg_train.pickle"),
#                                    os.path.join(path_library,
#                                                 f"neg_library_GNPS_{file_date}"))
#
# file_creator.create_all_library_files(os.path.join(path_data,
#                                                    f"GNPS_{file_date}_neg_tanimoto_scores.pickle"),
#                                       os.path.join(path_library,
#                                                    f"ms2ds_GNPS_{file_date}.hdf5"),
#                                       os.path.join(path_library,
#                                                    f"neg_mode_spec2vec_model_GNPS_{file_date}.model")
#                                       )
