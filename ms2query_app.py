import streamlit as st
from ms2query.app_helpers import get_query
from ms2query.app_helpers import get_library_data
from ms2query.app_helpers import make_downloads_folder
from ms2query.app_helpers import get_model
from ms2query.app_helpers import do_spectrum_processing
from ms2query.app_helpers import get_example_library_matches
from ms2query.app_helpers import get_library_matches
from ms2query.app_helpers import make_network_plot
from ms2query.app_helpers import get_library_similarities


logo_html = '<img src="https://github.com/iomega/ms2query/raw/main/images/ms2query_logo.svg" width="300">'
st.markdown(
    logo_html, unsafe_allow_html=True,
)

st.sidebar.markdown("## Select data and model")

st.write("""
Upload your query spectrum, and choose a spectrum library and Spec2Vec model in
the sidebar. Query the library and inspect the proposed results!
""")
st.write("## Input information")
input_warning_placeholder = st.empty()  # input warning for later

# load query spectrum
query_spectrums = get_query()
# get the download folder which is user adjustable
downloads_folder = make_downloads_folder()
# load library file in sidebar
library_spectrums, lib_is_processed, lib_num = get_library_data(
    downloads_folder)

# load a s2v model in sidebar
model, model_num = get_model(downloads_folder)

# write an input warning
if not query_spectrums or not library_spectrums or not model:
    input_warning_placeholder.markdown("""<p><span style="color:red">Please
    choose a query, library and model in the sidebar.</span></p>""",
                                       unsafe_allow_html=True)

# processing of query and library spectra into SpectrumDocuments
documents_query, documents_library = do_spectrum_processing(query_spectrums,
                                                            library_spectrums,
                                                            lib_is_processed)

# do library matching
st.write("## Library matching")
# load example library matching (test query on test library)
get_example_library_matches()

do_library_matching = st.checkbox("Do library matching")
if do_library_matching:
    if all([documents_query, documents_library, model]):
        found_match = get_library_matches(documents_query, documents_library,
                                          model, lib_num, model_num)
    else:
        do_library_matching = False
        st.write("""<p><span style="color:red">Please specify input files.
        </span></p>""", unsafe_allow_html=True)

# do networking
st.write("## Networking")
st.write("""This will trigger a large download if library similarities are not
yet present in the download folder.""")
plot_true = st.checkbox("""Plot network of found matches""")
if plot_true and do_library_matching:
    sim_matrix = get_library_similarities(
        found_match, documents_library, lib_num, downloads_folder)
    if sim_matrix is not None:
        make_network_plot(found_match, documents_library, sim_matrix)
    else:
        st.write("Similarity matrix not yet implemented for this library.")
elif plot_true:  # library matching is not done yet, but plot button is clicked
    st.write("""<p><span style="color:red">Please specify input files and do
            library matching.</span></p>""", unsafe_allow_html=True)

logo_html = '<img src="https://github.com/iomega/ms2query/raw/main/images/ms2query_logo.svg" width="200">'
st.sidebar.markdown(
    logo_html, unsafe_allow_html=True,
)

# Sidebar footer
sidebar_footer = """
Code: [MS2Query on GitHub](https://github.com/iomega/ms2query)  
Developed by Joris Louwen, Justin JJ van der Hooft, Florian Huber  
References:
[matchms](https://github.com/matchms/matchms)
[Spec2Vec](https://github.com/iomega/spec2vec)
"""
st.sidebar.markdown(sidebar_footer)
