import os
from typing import Tuple, List, Union
import pandas as pd
import streamlit as st
from spec2vec import SpectrumDocument
from matchms.Spectrum import Spectrum
from gensim.models import Word2Vec
from gensim.models.basemodel import BaseTopicModel
from urllib.request import urlretrieve
from ms2query.utils import json_loader
from ms2query.s2v_functions import process_spectrums
from ms2query.s2v_functions import set_spec2vec_defaults
from ms2query.s2v_functions import library_matching
from ms2query.networking import do_networking


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


def get_library_data() -> Tuple[List[Spectrum], Union[pd.DataFrame, None]]:
    """
    Return library, library_similarities as ([Spectrum], df) from user input
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

    # load similarity matrix, not implemented yet apart from test sim matrix
    if library_example == 'testspectrum_library.json' and not library_file:
        test_sim_matrix_file = os.path.join(
            os.path.split(os.path.dirname(__file__))[0], "tests",
            "test_found_matches_similarity_matrix.csv")
        test_sim_matrix = pd.read_csv(test_sim_matrix_file, index_col=0)
    else:
        test_sim_matrix = None
        st.write("""<p><span style="color:red">Libraries other than the example
            testspectrum are not implemented yet, so network plotting will not
            work for this library.</span></p>""", unsafe_allow_html=True)

    return library_spectrums, test_sim_matrix


def get_model() -> Tuple[Union[Word2Vec, None], Union[int, None]]:
    """Return (Word2Vec model, model number) and print some info in the app

    Models will be downloaded from zenodo to ../ms2query/downloads
    """
    st.write("#### Spec2Vec model")
    # get all data from zenodo
    base_dir = os.path.split(os.path.dirname(__file__))[0]
    downloads = os.path.join(base_dir, "downloads")
    model_name, model_file, model_num = get_zenodo_models(downloads)
    model = None
    if model_name:
        st.write("Your selected model:", model_name)
        model = Word2Vec.load(model_file)
    return model, model_num


def get_zenodo_models_dict(output_folder: str):
    """Get all urls and file locations for the downloadable models in the app

    Args:
    -------
    output_folder:
        Folder to download to
    """
    # configure all model link/name info
    all_pos_urls = [
        "https://zenodo.org/record/4173596/files/spec2vec_AllPositive_ratio0" +
        "5_filtered_201101_iter_15.model?download=1", "https://zenodo.org/re" +
        "cord/4173596/files/spec2vec_AllPositive_ratio05_filtered_201101_ite" +
        "r_15.model.trainables.syn1neg.npy?download=1", "https://zenodo.org/" +
        "record/4173596/files/spec2vec_AllPositive_ratio05_filtered_201101_i" +
        "ter_15.model.wv.vectors.npy?download=1"]
    removed_4000_urls = [
        "https://zenodo.org/record/4277395/files/spec2vec_library_testing_40" +
        "00removed_2dec.model?download=1", "https://zenodo.org/record/427739" +
        "5/files/spec2vec_library_testing_4000removed_2dec.model.trainables." +
        "syn1neg.npy?download=1", "https://zenodo.org/record/4277395/files/s" +
        "pec2vec_library_testing_4000removed_2dec.model.wv.vectors.npy?downl" +
        "oad=1"]
    all_pos_files = url_to_file(all_pos_urls, output_folder)
    removed_4000_files = url_to_file(removed_4000_urls, output_folder)
    model_dict = {"AllPositive model": (all_pos_urls, all_pos_files, 0),
                  "Case study 4000 removed spectra": (removed_4000_urls,
                                                      removed_4000_files, 1)}
    return model_dict


def url_to_file(all_urls: List[str], output_folder: str) -> List[str]:
    """Turn list of urls into list of files in output_folder

    Args:
    -------
    all_urls:
        The zenodo urls to transform the files in output_folder
    output_folder
    """
    all_files = []
    for url_name in all_urls:
        url_out_name = os.path.split(url_name)[-1].rpartition("?download")[0]
        out_path = os.path.join(output_folder, url_out_name)
        all_files.append(out_path)
    return all_files


def get_zenodo_models(output_folder: str = "downloads") -> Tuple[str, str,
                                                                 int]:
    """Returns list of file_paths to the downloaded files in order of input

    Args:
    -------
    output_folder:
        Folder to download to
    """
    model_dict = get_zenodo_models_dict(output_folder)
    model_name = st.sidebar.selectbox("Choose a Spec2Vec model",
                                      options=[""] + list(model_dict.keys()))
    model_file = None
    model_num = None
    if model_name:
        make_folder(output_folder)
        urls, files, model_num = model_dict[model_name]
        model_file = files[0]  # as it is first element in e.g. all_pos_urls
        for url_name, file_name in zip(urls, files):
            place_holder = st.empty()
            if not os.path.isfile(file_name):
                file_base = os.path.split(file_name)[-1]
                place_holder.write(f"Downloading {file_base} from zenodo...")
                urlretrieve(url_name, file_name)
                place_holder.write("Download successful.")
            place_holder.empty()
    return model_name, model_file, model_num


def make_folder(output_folder):
    """Create output_folder if it doesn't exist yet

    Args:
    -------
    output_folder
        Folder to create
    """
    if not os.path.isdir(output_folder):
        extend = os.path.split(output_folder)[-1]
        st.write(f"Writing downloaded files to {extend} directory")
        os.mkdir(output_folder)  # make output_folder if it doesn't exist


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
            outside [{settings["mz_from"]}, {settings["mz_to"]}] m/z window
            \n* remove spectra with < {settings["n_required"]} peaks\n* reduce
            number of peaks to maximum of {settings["ratio_desired"]} * parent
            mass\n* remove peaks with intensities <
            {settings["intensity_from"]} of maximum intensity (unless this
            brings number of peaks to less than 10)\n* add losses between m/z
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


def make_network_plot(found_match: pd.DataFrame,
                      documents_library: List[SpectrumDocument],
                      sim_matrix: pd.DataFrame):
    """Plots the network in the app

    Args:
    -------
    found_match
        Dataframe containing the scores of the library matches, index names
        should correspond to indices in documents_library
    documents_library
        Library spectra in SpectrumDocument format
    sim_matrix
        Dataframe containing the tanimoto similarities of the library spectra
        amongst each other
    """
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
    with col2:
        st.write("Restrict library connections")
        tanimoto_cutoff = st.slider("Tanimoto cutoff", min_value=0.,
                                    max_value=1., step=0.05, value=0.6)

    network_plot = do_networking("query", found_match, sim_matrix,
                                 documents_library, attribute_key=attr_key,
                                 cutoff=attr_cutoff,
                                 tan_cutoff=tanimoto_cutoff)
    if network_plot:
        plot_placeholder.plotly_chart(network_plot)
