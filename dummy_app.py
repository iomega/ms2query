import streamlit as st
from ms2query.utils import json_loader
import os

st.title("Query a Spec2Vec model and plot the results!")
st.write("""
Upload your query and library spectra files in json format.
Query the library using a Spec2Vec model and inspect the results! 
""")

# load query file in sidebar
query_file = st.sidebar.file_uploader("Choose a query spectrum file...",
                                      type=['json', 'txt'])
# gather default queries
test_query_file = os.path.join(os.path.dirname(__file__), 'tests',
                               'testspectrum_query.json')
example_queries_dict = {'testspectrum_query.json': test_query_file}
example_queries_list = [''] + list(example_queries_dict.keys())  # '' as default
query_example = st.sidebar.selectbox("Load a query spectrum example",
                                     example_queries_list)
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
    fig = query_spectrums[0].plot()
    st.pyplot(fig)

# load library file in sidebar
library_file = st.sidebar.file_uploader("Choose a spectra library file...",
                                        type=['json', 'txt'])
if library_file is not None:
    if library_file.name.endswith("json"):
        library_file.seek(0)  # fix for streamlit issue #2235
        library_spectrums = json_loader(library_file)
