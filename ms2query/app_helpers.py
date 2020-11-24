import os
import pickle
from typing import Tuple, List, Union, Dict
import numpy as np
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


def make_downloads_folder():
    """Return user adjustable download folder, default is ms2query/downloads
    """
    base_dir = os.path.split(os.path.dirname(__file__))[0]
    out_folder = os.path.join(base_dir, "downloads")
    different_out_folder = st.sidebar.text_input(
        "Change the download folder location. Default is: ms2query/downloads")
    if different_out_folder:
        out_folder = different_out_folder
    return out_folder


def get_library_data(output_dir: str) -> Tuple[List[Spectrum], bool, int]:
    """
    Return library, 'is lib processed', lib number as ([Spectrum], bool, int)

    Args:
    ------
    output_dir:
        Folder to download zenodo libraries to
    """
    library_spectrums = []  # default so later code doesn't crash
    lib_num = None
    # gather default libraries
    example_libs_dict, example_libs_list = gather_test_json(
        'testspectrum_library.json')
    zenodo_dict = gather_zenodo_library(output_dir)
    example_libs_dict.update(zenodo_dict)
    example_libs_list.extend(list(zenodo_dict.keys()))
    library_example = st.sidebar.selectbox("""Choose a spectrum library (will 
    trigger large download if library not present in download folder)""",
                                           example_libs_list)
    processed = bool(library_example in zenodo_dict.keys())

    st.write("#### Library spectra")
    if library_example:
        if processed:
            # download from zenodo
            file_name, lib_num = download_zenodo_library(
                example_libs_dict, library_example, output_dir)
            library_spectrums = load_pickled_file(file_name)
            if isinstance(library_spectrums, tuple):
                # in the case of the case study set that is tuple(query, lib)
                if lib_num == 1:  # the case study library
                    library_spectrums = library_spectrums[1]
                if lib_num == 2:
                    # add query and lib from case study: get AllPositive back
                    library_spectrums = library_spectrums[0] + \
                                        library_spectrums[1]
            st.write(f"Your selected library: {library_example}")
        else:
            st.write('You have selected the small test library:',
                     library_example)
            lib_num = 0
            library_spectrums = json_loader(
                open(example_libs_dict[library_example]))

    # write library info
    if library_spectrums:
        st.write(f"Your library contains {len(library_spectrums)} spectra.")
    return library_spectrums, processed, lib_num


def download_zenodo_library(example_libs_dict: Dict[str, Tuple[str, str, int]],
                            library_example: str,
                            output_dir: str) -> Tuple[str, int]:
    """Downloads the library from zenodo and returns the file_path and lib_num

    Args:
    -------
    example_libs_dict:
        Dict linking the library name in the app to the zenodo url, the path
        of where the library should be downloaded and the library number
    library_example
        The library name in the app that is chosen from user input
    output_dir
        Folder to download zenodo libraries to
    """
    make_folder(output_dir)
    url_name, file_name, lib_num = example_libs_dict[library_example]
    place_holder = st.empty()
    if not os.path.isfile(file_name):
        file_base = os.path.split(file_name)[-1]
        place_holder.write(
            f"Downloading {file_base} from zenodo...")
        urlretrieve(url_name, file_name)
        place_holder.write("Download successful.")
    place_holder.empty()
    return file_name, lib_num


@st.cache(allow_output_mutation=True)  # for speedup, e.a. lib is not mutated
def load_pickled_file(file_name: str):
    """Returns contents from the pickle file

    Args:
    -------
    file_name:
        Path of the pickle file to read
    """
    with open(file_name, "rb") as inf:
        contents = pickle.load(inf)
    return contents


def gather_zenodo_library(output_folder: str):
    """Gather file and url info for zenodo libraries

    Args:
    ------
    output_folder
        Folder to download the libraries to.
    """
    test_set_all_pos = ("https://zenodo.org/record/4281172/files/testing_que" +
                        "ry_library_s2v_2dec.pickle?download=1")
    test_set_all_pos_file = url_to_file([test_set_all_pos], output_folder)[0]
    library_dict = {
        "AllPositive dataset": (test_set_all_pos, test_set_all_pos_file, 2),
        "Case study AllPositive subset":
            (test_set_all_pos, test_set_all_pos_file, 1)}
    return library_dict


def get_model(out_folder: str) -> Tuple[Union[Word2Vec, None],
                                        Union[int, None]]:
    """Return (Word2Vec model, model number) and print some info in the app

    Models will be downloaded from zenodo to ../ms2query/downloads

    Args:
    -------
    out_folder:
        Folder of where to download the models
    """
    st.write("#### Spec2Vec model")
    # get all data from zenodo
    model_name, model_file, model_num = get_zenodo_models(out_folder)
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
    model_name = st.sidebar.selectbox("""Choose a Spec2Vec model (will trigger
    large download if model not present in download folder)""",
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
                           library_spectrums: List[Union[Spectrum,
                                                         SpectrumDocument]],
                           library_is_processed: bool) -> Tuple[
    List[SpectrumDocument], List[SpectrumDocument]]:
    """Process query, library into SpectrumDocuments and write processing info

    Args:
    -------
    query_spectrums:
        Query spectra in matchms.Spectrum format
    library_spectrums:
        Library spectra in matchms.Spectrum or SpectrumDocument format
    library_is_processed:
        Bool for telling if the library is already processed -> don't process
        it again. In this case the output equals input for library
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
    if library_is_processed:
        documents_library = library_spectrums
    else:
        documents_library = process_spectrums(library_spectrums, **settings)

    return documents_query, documents_library


def get_example_library_matches() -> pd.DataFrame:
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

    return test_found_matches


def get_library_matches(documents_query: List[SpectrumDocument],
                        documents_library: List[SpectrumDocument],
                        model: BaseTopicModel,
                        lib_num: int,
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
    documents_library_class = DocumentsLibrary(documents_library)
    found_matches_s2v = cached_library_matching(
        documents_query, documents_library_class, model, topn, lib_num,
        model_num)

    st.write("These are the library matches for your query")
    if found_matches_s2v:
        first_found_match = found_matches_s2v[0]
        first_found_match = first_found_match.sort_values(
            "s2v_score", ascending=False)
        st.dataframe(first_found_match.iloc[:show_topn])
        return first_found_match
    return None


class DocumentsLibrary:
    """Dummy class used to circumvent hashing the library for library matching
    """

    def __init__(self, documents: List[SpectrumDocument]):
        """

        Args:
        -------
        documents:
            Library spectra as SpectrumDocuments
        """
        self.documents = documents


@st.cache(hash_funcs={DocumentsLibrary: lambda _: None,
                      Word2Vec: lambda _: None})
def cached_library_matching(documents_query: List[SpectrumDocument],
                            documents_library: DocumentsLibrary,
                            model: BaseTopicModel,
                            topn: int,
                            lib_num: int,
                            model_num: int) -> List[pd.DataFrame]:
    """Run library matching for the app and cache the result with st.cache

    Returns the usual list of library matches as DataFrames

    Args:
    -------
    documents_query:
        Query spectra in SpectrumDocument format
    documents_library:
        Library spectra in DocumentsLibrary format
    model:
        A trained Spec2Vec model
    topn:
        The amount of Spec2Vec top candidates to retrieve
    lib_num:
        The library number used for library matching. This is a workaround for
        the caching of the library matches as the library is expensive to hash.
        Library is not hashed and with model_num it is kept into account if the
        library changes.
    model_num:
        The model number used for library matching. This is a workaround for
        the caching of the library matches as the model is expensive to hash.
        Model is not hashed and with model_num it is kept into account if the
        model changes.
    """
    # pylint: disable=too-many-arguments
    if lib_num:  # variable for the hash function
        pass
    if model_num:  # variable for the hash function
        pass
    documents_library = documents_library.documents
    found_matches_s2v = library_matching(
        documents_query, documents_library, model,
        presearch_based_on=[f"spec2vec-top{topn}", "parentmass"],
        **{"allowed_missing_percentage": 100})
    return found_matches_s2v


def get_library_similarities(found_match: pd.DataFrame,
                             documents_library: List[SpectrumDocument],
                             library_num: int,
                             output_folder: str = "downloads") -> np.array:
    """Returns sim matrix as np.array, index order corresponds to found_match

    Args:
    ------
    found_match:
        Dataframe containing the scores of the library matches
    documents_library:
        Library spectra in DocumentsLibrary format
    library_num:
        The library number, 0 means the example library, 1 and 2 (subset of)
        AllPositive library
    output_folder:
        Location to download/get similarity matrix and metadata from
    """
    # pylint: disable=protected-access
    if library_num == 0:
        test_sim_matrix_file = os.path.join(
            os.path.split(os.path.dirname(__file__))[0], "tests",
            "test_found_matches_similarity_matrix.csv")
        return np.array(pd.read_csv(test_sim_matrix_file, index_col=0))
    if library_num in (1, 2):
        # construct the slice of the similarity matrix in order of matches ind
        # take 100 as a max value, same as in library_matching
        match_inds = found_match.iloc[:100].index.to_list()
        match_inchi14 = [documents_library[ind]._obj.get("inchikey")[:14]
                         for ind in match_inds]
        sim_slice_inds = get_sim_matrix_lookup(match_inchi14, output_folder)
        return subset_sim_matrix(sim_slice_inds, output_folder)

    return None


def get_sim_matrix_lookup(match_inchi14: List[str],
                          output_folder: str = "downloads") -> List[int]:
    """Return list of indices pointing to a row/col in the similarity matrix

    The metadata file is opened and the indices are extracted in order of the
    input inchi14s

    Args:
    ------
    match_inchi14
        List of the first 14 chars of inchikeys for the library matches in
        order of occurrence in the match df
    output_folder:
        Location to download/get similarity matrix and metadata from
    """
    metadata_url = "https://zenodo.org/record/4286949/files/metadata_AllIn" + \
                   "chikeys14.csv?download=1"
    metadata_name = "metadata_AllInchikeys14.csv"
    metadata_file = os.path.join(output_folder, metadata_name)
    if not os.path.exists(metadata_file):
        place_holder = st.empty()
        place_holder.write(f"Downloading {metadata_name} from zenodo...")
        urlretrieve(metadata_url, metadata_file)
        place_holder.empty()

    with open(metadata_file, 'r') as inf:
        inf.readline()
        inchi_dict = {}
        for line in inf:
            line = line.strip().split(',')
            inchi_dict[line[1]] = int(line[0])  # dict{inchi: index_lookup}

    indices = [inchi_dict[inchi] for inchi in match_inchi14]
    return indices


def subset_sim_matrix(indices: List[int],
                      output_folder: str = "downloads") -> np.array:
    """Returns sim matrix subset of indices vs indices in order

    Accesses sim matrix from disk

    Args:
    -------
    indices:
        In order, the indices for creating the subset of the sim matrix
    output_folder:
        Location to download/get similarity matrix and metadata from
    """
    sim_url = "https://zenodo.org/record/4286949/files/similarities_AllInch" + \
              "ikeys14_daylight2048_jaccard.npy?download=1"
    sim_name = "similarities_AllInchikeys14_daylight2048_jaccard.npy"
    sim_file = os.path.join(output_folder, sim_name)
    if not os.path.exists(sim_file):
        place_holder = st.empty()
        place_holder.write(f"Downloading {sim_name} from zenodo...")
        urlretrieve(sim_url, sim_file)
        place_holder.empty()

    sim_map = np.lib.format.open_memmap(sim_file, dtype="float64", mode="r")
    row_slice = np.take(sim_map, indices, 0)
    final_slice = np.take(row_slice, indices, 1)

    return np.array(final_slice)


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
    # pylint: disable=too-many-locals
    def_show_topn = 10  # default for topn results to show
    if len(documents_library) < def_show_topn:
        def_show_topn = len(documents_library)
    cols = st.beta_columns([1, 3])
    with cols[0]:
        show_topn = int(st.text_input("Show top n matches (1-100)",
                                      value=def_show_topn))
    found_match = found_match.iloc[:show_topn]

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
