import os
import pandas as pd
import streamlit as st
from gensim.models import Word2Vec
from ms2query.utils import json_loader
from ms2query.s2v_functions import set_spec2vec_defaults
from ms2query.s2v_functions import process_spectrums
from ms2query.s2v_functions import library_matching
from ms2query.networking import do_networking
from ms2query.networking import plotly_network

st.title("Ms2query")
st.write("""
Upload your query and library spectra files in json format in the sidebar.
Query the library using a Spec2Vec model and inspect the results! 
""")

# load query file in sidebar
query_spectrums = []  # default so later code doesn't crash
query_file = st.sidebar.file_uploader("Choose a query spectrum file...",
                                      type=['json', 'txt'])
# gather default queries
test_query_file = os.path.join(os.path.dirname(__file__), 'tests',
                               'testspectrum_query.json')
example_queries_dict = {'testspectrum_query.json': test_query_file}
example_queries_list = [''] + list(example_queries_dict.keys())
query_example = st.sidebar.selectbox("Load a query spectrum example",
                                     example_queries_list)

st.write("## Input information")
input_warning_placeholder = st.empty()  # input warning for later
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

# load library file in sidebar
library_spectrums = []  # default so later code doesn't crash
library_file = st.sidebar.file_uploader("Choose a spectra library file...",
                                        type=['json', 'txt'])
# gather default libraries
test_library_file = os.path.join(os.path.dirname(__file__), 'tests',
                                 'testspectrum_library.json')
example_libs_dict = {'testspectrum_library.json': test_library_file}
example_libs_list = [''] + list(example_libs_dict.keys())  # '' as default
library_example = st.sidebar.selectbox("Load a library spectrum example",
                                       example_libs_list)
st.write("#### Library spectra")
if library_example:
    st.write('You have selected an example library:', library_example)
    library_spectrums = json_loader(open(example_libs_dict[library_example]))
elif library_file is not None:
    if library_file.name.endswith("json"):
        library_file.seek(0)  # fix for streamlit issue #2235
        library_spectrums = json_loader(library_file)

# write library info
if library_spectrums:
    st.write(f"Your library contains {len(library_spectrums)} spectra.")

# load a s2v model in sidebar
# todo: make more user friendly, currently there is no standard func to do this
# for quick testing C:\Users\joris\Documents\eScience_data\data\trained_models\spec2vec_library_testing_4000removed_2dec.model
model_file = st.sidebar.text_input(
    "Enter filename of Spec2Vec model (with path):")
st.write("#### Spec2Vec model")
model = None
if model_file:
    if model_file.endswith(".model"):
        st.write("Your selected model:", os.path.split(model_file)[-1])
        model = Word2Vec.load(model_file)
    else:
        st.write("""<p><span style="color:red">Model file extension should be
        .model, please try again.</span></p>""", unsafe_allow_html=True)

# write an input warning
if not query_spectrums or not library_spectrums or not model_file:
    input_warning_placeholder.markdown("""<p><span style="color:red">Please
    upload a query, library and model file in the sidebar.</span></p>""",
                                       unsafe_allow_html=True)

# processing of query and library spectra into SpectrumDocuments
st.write("""## Post-process spectra
Spec2Vec similarity scores rely on creating a document vector for each
spectrum. For the underlying word2vec model we want the documents (=spectra) to
be more homogeneous in their number of unique words. Assuming that larger
compounds will on average break down into a higher number of meaningful
fragment peaks we reduce the document size of each spectrum according to its
parent mass.
""")
# todo: we could add some buttons later to make this adjustable
settings = set_spec2vec_defaults()
with st.beta_expander("View processing defaults"):
    st.markdown(f"""* normalize peaks (maximum intensity to 1)\n* remove peaks 
    outside [{settings["mz_from"]}, {settings["mz_to"]}] m/z window\n* remove
    spectra with < {settings["n_required"]} peaks\n* reduce number of peaks to
    maximum of {settings["ratio_desired"]} * parent mass\n* remove peaks with
    intensities < {settings["intensity_from"]} of maximum intensity (unless
    this brings number of peaks to less than 10)\n* add losses between m/z
    value of [{settings["loss_mz_from"]}, {settings["loss_mz_to"]}]""")

documents_query = process_spectrums(query_spectrums, **settings)
documents_library = process_spectrums(library_spectrums, **settings)

# do library matching
# for now load example as library matching function isn't there yet
path_dir = os.path.dirname(__file__)
test_found_matches_file = os.path.join(path_dir, "tests",
                                       "test_found_matches.csv")
test_found_matches = pd.read_csv(test_found_matches_file, index_col=0)
st.write("## Library matching")
with st.beta_expander("See an example"):
    st.write("These are the test library matches for test query:")
    st.dataframe(test_found_matches)

# library matching function
do_library_matching = st.checkbox("Do library matching")
if do_library_matching:
    if all([model, documents_query, documents_library]):
        if len(documents_library) < 20:
            def_topn = len(documents_library)
        else:
            def_topn = 20
        cols = st.beta_columns([1, 4])
        with cols[0]:
            topn = st.text_input("Show top n matches", value=def_topn)
        st.write("These are the library matches for your query")
        found_matches_s2v = library_matching(
            documents_query, documents_library, model, presearch_based_on=[
                f"spec2vec-top{topn}", "parentmass"],
            **{"allowed_missing_percentage": 100})
        if found_matches_s2v:
            found_match = found_matches_s2v[0]
            st.dataframe(found_match.sort_values("s2v_score", ascending=False))
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
    plot_placeholder = st.empty()  # add a place for the plot
    # add sliders to adjust network plot
    col1, col2 = st.beta_columns(2)
    with col1:
        st.write("Restrict library matches")
        attr_key = st.selectbox("Choose parameter", found_match.columns,
                                index=0)
        attr_data = found_match[attr_key]
        if isinstance(attr_data.iloc[0], float):
            # true for s2v, cosine etc
            min_v, max_v, step, val = (0., 1., 0.05, 0.4)
        elif max(attr_data) >= 1:
            # true for parentmass, cosine matches etc
            min_v, max_v, step, val = (0, max(attr_data), 1, 1)
        attr_cutoff = st.slider(attr_key + " cutoff", min_value=min_v,
                                max_value=max_v, step=step, value=val)
        draw_edge_labels = st.checkbox("Show values")
    with col2:
        st.write("Restrict library connections")
        tanimoto_cutoff = st.slider("Tanimoto cutoff", min_value=0.,
                                    max_value=1., step=0.05, value=0.6)

    network, node_labels = do_networking("query", found_match, test_sim_matrix,
                                         documents_library)
    network_plot = plotly_network(network, node_labels, attribute_key=attr_key,
                                  cutoff=attr_cutoff,
                                  tan_cutoff=tanimoto_cutoff)
    if network_plot:
        plot_placeholder.plotly_chart(network_plot)
elif plot_true:  # library matching is not done yet, but plot button is clicked
    st.write("""<p><span style="color:red">Please specify input files and do
            library matching.</span></p>""", unsafe_allow_html=True)
