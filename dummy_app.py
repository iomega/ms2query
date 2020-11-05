import streamlit as st
from ms2query.utils import json_loader

st.title("Query a Spec2Vec model and plot the results!")
st.write("""
Upload your query and library spectra files in json format.
Query the library using a Spec2Vec model and inspect the results! 
""")

# load query file in sidebar
query_file = st.sidebar.file_uploader("Choose a query spectrum file...",
                                      type=['json', 'txt'])
if query_file is not None:
    if query_file.name.endswith("json"):
        query_file.seek(0)  # fix for streamlit issue #2235
        query_spectrums = json_loader(query_file)
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
