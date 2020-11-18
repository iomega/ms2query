import os
from typing import Tuple, List, Union
import pandas as pd
import streamlit as st
from spec2vec import SpectrumDocument
from matchms.Spectrum import Spectrum
from gensim.models import Word2Vec
from gensim.models.basemodel import BaseTopicModel
from ms2query.utils import json_loader
from ms2query.s2v_functions import process_spectrums
from ms2query.s2v_functions import set_spec2vec_defaults
from ms2query.s2v_functions import library_matching


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


def get_query() -> List[Spectrum]:
    """
    Return query spectra as [Spectrum] from user input and print query info
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
        query_spectrums = json_loader(
            open(example_queries_dict[query_example]))
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


def get_library() -> List[Spectrum]:
    """
    Return library spectra as [Spectrum] from user input and print library info
    """
    library_spectrums = []  # default so later code doesn't crash
    library_file = st.sidebar.file_uploader("Choose a spectra library file...",
                                            type=['json', 'txt'])
    # gather default libraries
    example_libs_dict, example_libs_list = gather_test_json(
        'testspectrum_library.json')
    library_example = st.sidebar.selectbox("Load a library spectrum example",
                                           example_libs_list)
    st.write("#### Library spectra")
    if library_example and not library_file:
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


def get_model() -> Tuple[Union[Word2Vec, None], Union[int, None]]:
    """Return (Word2Vec model, model number) and print some info in the app
    """
    model_file = st.sidebar.text_input(
        "Enter filename of Spec2Vec model (with path):")
    st.write("#### Spec2Vec model")
    model = None
    model_num = None
    if model_file:
        if model_file.endswith(".model"):
            st.write("Your selected model:", os.path.split(model_file)[-1])
            model = Word2Vec.load(model_file)
            # todo: change this when multiple models are added for caching
            model_num = 0
        else:
            st.write("""<p><span style="color:red">Model file extension should
            be .model, please try again.</span></p>""", unsafe_allow_html=True)
    return model, model_num


def do_spectrum_processing(query_spectrums: List[Spectrum],
                           library_spectrums: List[Spectrum]) -> Tuple[
                           List[SpectrumDocument], List[SpectrumDocument]]:
    """Process query, library into SpectrumDocuments and write processing info

    Args:
    -------
    query_spectrums:
        Query spectra in matchms.Spectrum format
    library_spectrums:
        Library spectra in matchms.Spectrum format
    """
    st.write("## Post-process spectra")
    st.write("""Spec2Vec similarity scores rely on creating a document vector
    for each spectrum. For the underlying word2vec model we want the documents
    (=spectra) to be more homogeneous in their number of unique words. Assuming
    that larger compounds will on average break down into a higher number of
    meaningful fragment peaks we reduce the document size of each spectrum
    according to its parent mass.
    """)
    # todo: we could add some buttons later to make this adjustable
    settings = set_spec2vec_defaults()
    with st.beta_expander("View processing defaults"):
        st.markdown(
            f"""* normalize peaks (maximum intensity to 1)\n* remove peaks 
        outside [{settings["mz_from"]}, {settings["mz_to"]}] m/z window\n* remove
        spectra with < {settings["n_required"]} peaks\n* reduce number of peaks to
        maximum of {settings["ratio_desired"]} * parent mass\n* remove peaks with
        intensities < {settings["intensity_from"]} of maximum intensity (unless
        this brings number of peaks to less than 10)\n* add losses between m/z
        value of [{settings["loss_mz_from"]}, {settings["loss_mz_to"]}]""")

    documents_query = process_spectrums(query_spectrums, **settings)
    documents_library = process_spectrums(library_spectrums, **settings)

    return documents_query, documents_library


def get_example_library_matches():
    """
    Get test_found_matches from test dir, display it as an example in the app
    """
    base_dir = os.path.split(os.path.dirname(__file__))[0]
    test_found_matches_file = os.path.join(base_dir, "tests",
                                           "test_found_matches.csv")
    test_found_matches = pd.read_csv(test_found_matches_file, index_col=0)
    with st.beta_expander("See an example"):
        st.write("These are the test library matches for test query:")
        st.dataframe(test_found_matches)


def get_library_matches(documents_query: List[SpectrumDocument],
                        documents_library: List[SpectrumDocument],
                        model: BaseTopicModel,
                        model_num: int) -> Union[pd.DataFrame, None]:
    """Returns DataFrame of library matches for first query in documents_query

    Args:
    -------
    documents_query:
        Query spectra in SpectrumDocument format
    documents_library
        Library spectra in SpectrumDocument format
    model
        A trained Spec2Vec model
    model_num
        The model number used for library matching. This is a workaround for
        the caching of the library matches as the model is expensive to hash,
        it is not hashed and with model_num it is kept into account if the
        model changes.
    """
    topn = 100  # assume that user will never want to see more than 100 matches
    if len(documents_library) < topn:
        topn = len(documents_library)  # so it doesn't crash with small libs

    def_show_topn = 20  # default for topn results to show
    if len(documents_library) < def_show_topn:
        def_show_topn = len(documents_library)
    cols = st.beta_columns([1, 4])
    with cols[0]:
        show_topn = int(st.text_input("Show top n matches",
                                      value=def_show_topn))

    st.write("These are the library matches for your query")
    found_matches_s2v = cached_library_matching(
        documents_query, documents_library, model, topn, model_num)

    if found_matches_s2v:
        first_found_match = found_matches_s2v[0]
        st.dataframe(first_found_match.sort_values(
            "s2v_score", ascending=False).iloc[:show_topn])
        return first_found_match
    return None


@st.cache(hash_funcs={Word2Vec: lambda _: None})
def cached_library_matching(documents_query: List[SpectrumDocument],
                            documents_library: List[SpectrumDocument],
                            model: BaseTopicModel,
                            topn: int,
                            model_num: int) -> List[pd.DataFrame]:
    """Run library matching for the app and cache the result with st.cache

    Returns the usual list of library matches as DataFrames

    Args:
    -------
    documents_query:
        Query spectra in SpectrumDocument format
    documents_library:
        Library spectra in SpectrumDocument format
    model:
        A trained Spec2Vec model
    topn:
        The amount of Spec2Vec top candidates to retrieve
    model_num
        The model number used for library matching. This is a workaround for
        the caching of the library matches as the model is expensive to hash,
        it is not hashed and with model_num it is kept into account if the
        model changes.
    """
    if model_num:  # variable for the hash function
        pass
    found_matches_s2v = library_matching(
        documents_query, documents_library, model,
        presearch_based_on=[f"spec2vec-top{topn}", "parentmass"],
        **{"allowed_missing_percentage": 100})
    return found_matches_s2v
