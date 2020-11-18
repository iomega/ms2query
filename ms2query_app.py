import os
import pandas as pd
import streamlit as st
from ms2query.app_helpers import get_query
from ms2query.app_helpers import get_library
from ms2query.app_helpers import get_model
from ms2query.app_helpers import do_spectrum_processing
from ms2query.app_helpers import get_example_library_matches
from ms2query.app_helpers import get_library_matches
from ms2query.app_helpers import make_network_plot


st.title("Ms2query")
st.write("""
Upload your query and library spectra files in json format in the sidebar.
Query the library using a Spec2Vec model and inspect the results! 
""")
st.write("## Input information")
input_warning_placeholder = st.empty()  # input warning for later

# load query spectrum
query_spectrums = get_query()
# load library file in sidebar
library_spectrums = get_library()

# load a s2v model in sidebar
# todo: make more user friendly, currently there is no standard func to do this
# for quick testing C:\Users\joris\Documents\eScience_data\data\trained_models\spec2vec_library_testing_4000removed_2dec.model
model, model_num = get_model()

# write an input warning
if not query_spectrums or not library_spectrums or not model:
    input_warning_placeholder.markdown("""<p><span style="color:red">Please
    upload a query, library and model file in the sidebar.</span></p>""",
                                       unsafe_allow_html=True)

# processing of query and library spectra into SpectrumDocuments
documents_query, documents_library = do_spectrum_processing(query_spectrums,
                                                            library_spectrums)

# do library matching
st.write("## Library matching")
# load example library matching (test query on test library)
get_example_library_matches()

# library matching function
do_library_matching = st.checkbox("Do library matching")
if do_library_matching:
    if all([documents_query, documents_library, model]):
        found_match = get_library_matches(documents_query, documents_library,
                                          model, model_num)
    else:
        do_library_matching = False
        st.write("""<p><span style="color:red">Please specify input files.
        </span></p>""", unsafe_allow_html=True)

# do networking
# for now load example similarity matrix
path_dir = os.path.dirname(__file__)
test_sim_matrix_file = os.path.join(path_dir, "tests", "test_found_matches_" +
                                    "similarity_matrix.csv")
test_sim_matrix = pd.read_csv(test_sim_matrix_file, index_col=0)
st.write("## Networking")
plot_true = st.checkbox("Plot network of found matches")
if plot_true and do_library_matching:
    make_network_plot(found_match, documents_library, test_sim_matrix)
elif plot_true:  # library matching is not done yet, but plot button is clicked
    st.write("""<p><span style="color:red">Please specify input files and do
            library matching.</span></p>""", unsafe_allow_html=True)
