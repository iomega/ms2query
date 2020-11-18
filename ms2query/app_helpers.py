import os
import streamlit as st
from .utils import json_loader


def get_query():
    """Gather all relevant query information and print info for query spectrum
    """
    # load query file in sidebar
    query_spectrums = []  # default so later code doesn't crash
    query_file = st.sidebar.file_uploader("Choose a query spectrum file...",
                                          type=['json', 'txt'])
    # gather default queries
    test_query_file = os.path.join(os.path.split(os.path.dirname(__file__))[0],
                                   'tests', 'testspectrum_query.json')
    example_queries_dict = {'testspectrum_query.json': test_query_file}
    example_queries_list = [''] + list(example_queries_dict.keys())
    query_example = st.sidebar.selectbox("Load a query spectrum example",
                                         example_queries_list)

    st.write("#### Query spectrum")
    if query_example:
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
