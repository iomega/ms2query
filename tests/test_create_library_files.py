
# todo very old version has to be updated
# def test_store_ms2ds_embeddings(tmp_path):
#     path_to_test_files_dir = os.path.join(
#         os.path.split(os.path.dirname(__file__))[0],
#         'tests/test_files')
#     pickled_file_loc = os.path.join(path_to_test_files_dir,
#                                     "first_10_spectra.pickle")
#     ms2ds_model_file_loc = os.path.join(
#         path_to_test_files_dir,
#         "ms2ds_siamese_210207_ALL_GNPS_positive_L1L2.hdf5")
#     new_embeddings_file_name = os.path.join(tmp_path,
#                                             "test_ms2ds_embeddings.pickle")
#     control_embeddings_file_name = os.path.join(path_to_test_files_dir,
#                                                 "test_ms2ds_embeddings.pickle")
#     # Load spectra and model in memory
#     with open(pickled_file_loc, "rb") as pickled_file:
#         spectrum_list = pickle.load(pickled_file)
#     ms2ds_model = load_ms2ds_model(ms2ds_model_file_loc)
#
#     # Create new pickled embeddings file
#     store_ms2ds_embeddings(spectrum_list,
#                            ms2ds_model,
#                            new_embeddings_file_name)
#
#     # Open new pickled embeddings file
#     with open(new_embeddings_file_name, "rb") as new_embeddings_file:
#         embeddings = pickle.load(new_embeddings_file)
#     # Open control pickled embeddings file
#     with open(control_embeddings_file_name, "rb") as control_embeddings_file:
#         control_embeddings = pickle.load(control_embeddings_file)
#     # Test if the correct embeddings are loaded into the new file.
#     pd.testing.assert_frame_equal(embeddings, control_embeddings)
#
# def test_store_s2v_embeddings(tmp_path):
#     path_to_test_files_dir = os.path.join(
#         os.path.split(os.path.dirname(__file__))[0],
#         'tests/test_files')
#     pickled_file_loc = os.path.join(path_to_test_files_dir,
#                                     "first_10_spectra.pickle")
#     s2v_model_file_loc = os.path.join(
#         path_to_test_files_dir,
#         "spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model")
#     new_embeddings_file_name = os.path.join(tmp_path,
#                                             "test_embeddings_file.pickle")
#     control_embeddings_file_name = os.path.join(path_to_test_files_dir,
#                                                 "test_embeddings_file.pickle")
#     # Load spectra and model in memory
#     with open(pickled_file_loc, "rb") as pickled_file:
#         spectrum_list = pickle.load(pickled_file)
#     s2v_model = Word2Vec.load(s2v_model_file_loc)
#
#     # Create new pickled embeddings file
#     store_s2v_embeddings(spectrum_list,
#                          s2v_model,
#                          new_embeddings_file_name)
#     # Open new pickled embeddings file
#     with open(new_embeddings_file_name, "rb") as new_embeddings_file:
#         embeddings = pickle.load(new_embeddings_file)
#     # Open control pickled embeddings file
#     with open(control_embeddings_file_name, "rb") as control_embeddings_file:
#         control_embeddings = pickle.load(control_embeddings_file)
#     # Test if the correct embeddings are loaded into the new file.
#     pd.testing.assert_frame_equal(embeddings, control_embeddings)