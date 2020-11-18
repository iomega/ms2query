import os
from typing import Tuple
import streamlit as st
from ms2query.utils import json_loader


def gather_test_json(test_file_name: str) -> Tuple[dict, list]:
    """Return tuple of {test_file_name: full_path}, ['', test_file_name]

    test_file_name has to be in 'tests' folder

    Args:
    -------
    test_file_name:
        Name of the test file.
    """
    test_path = os.path.join(os.path.split(os.path.dirname(__file__))[0],
                             'tests', test_file_name)
    test_dict = {test_file_name: test_path}
    test_list = [''] + list(test_dict.keys())
    return test_dict, test_list


def get_query():
    """Gather all relevant query information and print info for query spectrum
    """
    # load query file in sidebar
    query_spectrums = []  # default so later code doesn't crash
    query_file = st.sidebar.file_uploader("Choose a query spectrum file...",
                                          type=['json', 'txt'])

    # gather default queries
    example_queries_dict, example_queries_list = gather_test_json(
        'testspectrum_query.json')
    query_example = st.sidebar.selectbox("Load a query spectrum example",
                                         example_queries_list)

    st.write("#### Query spectrum")
    if query_example and not query_file:
        st.write('You have selected an example query:', query_example)
        query_spectrums = json_loader(open(example_queries_dict[query_example]))
    elif query_file is not None:
        if query_file.name.endswith("json"):
            query_file.seek(0)  # fix for streamlit issue #2235
            query_spectrums = json_loader(query_file)

    # write query info
    if query_example or query_file:
        st.write("Your query spectrum id: {}".format(
            query_spectrums[0].metadata.get("spectrum_id")))
        with st.beta_expander("View additional query information"):
            fig = query_spectrums[0].plot()
            st.pyplot(fig)

    return query_spectrums


def get_library():
    library_spectrums = []  # default so later code doesn't crash
    library_file = st.sidebar.file_uploader("Choose a spectra library file...",
                                            type=['json', 'txt'])
    # gather default libraries
    example_libs_dict, example_libs_list = gather_test_json(
        'testspectrum_library.json')
    library_example = st.sidebar.selectbox("Load a library spectrum example",
                                           example_libs_list)
    st.write("#### Library spectra")
    if library_example:
        st.write('You have selected an example library:', library_example)
        library_spectrums = json_loader(
            open(example_libs_dict[library_example]))
    elif library_file is not None:
        if library_file.name.endswith("json"):
            library_file.seek(0)  # fix for streamlit issue #2235
            library_spectrums = json_loader(library_file)

    # write library info
    if library_spectrums:
        st.write(f"Your library contains {len(library_spectrums)} spectra.")

    return library_spectrums
