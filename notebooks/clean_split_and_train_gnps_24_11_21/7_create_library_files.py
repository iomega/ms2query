import os
from ms2query.library_files_creator import LibraryFilesCreator

path_root = os.path.dirname(os.getcwd())
path_data = os.path.join(os.path.dirname(path_root), "data/gnps_24_11_2021/positive_mode/")
file_creator = LibraryFilesCreator(os.path.join(path_data, "GNPS_24_11_2021_pos_train.pickle"),
                                   os.path.join(path_data, "GNPS_24_11_2021"))

file_creator.create_all_library_files(os.path.join(path_data, "GNPS_24_11_2021_pos_tanimoto_scores.pickle"),
                                      os.path.join(path_data, "trained_models/ms2ds_GNPS_25_11_2021.hdf5"),
                                      os.path.join(path_data, "spec2vec_models/spec2vec_model_GNPS_24_11_2021.model")
                                      )
