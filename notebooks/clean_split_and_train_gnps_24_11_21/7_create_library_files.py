import os
from ms2query.library_files_creator import LibraryFilesCreator

file_date = "15_12_2021"
path_data = "C:\\HSD\\OneDrive - Hochschule Düsseldorf\\Data\\ms2query"
path_library = os.path.join(os.path.dirname(path_data), "library_gnps_15_12")


file_creator = LibraryFilesCreator(os.path.join(path_data,
                                                f"GNPS_{file_date}_pos_train.pickle"),
                                   os.path.join(path_library,
                                                f"GNPS_{file_date}"))

file_creator.create_all_library_files(os.path.join(path_data,
                                                   f"GNPS_{file_date}_pos_tanimoto_scores.pickle"),
                                      os.path.join(path_data,
                                                   f"trained_models/ms2ds_GNPS_{file_date}.hdf5"),
                                      os.path.join(path_data,
                                                   f"spec2vec_models/spec2vec_model_GNPS_{file_date}.model")
                                      )
