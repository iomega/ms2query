from typing import List
from typing import Union
import numpy as np
import pandas as pd
from gensim.models.basemodel import BaseTopicModel
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity
from matchms.filtering import reduce_to_number_of_peaks
from matchms.filtering import add_losses
from matchms.similarity import CosineGreedy, ModifiedCosine, ParentMassMatch
from spec2vec import SpectrumDocument
from spec2vec import Spec2Vec


# pylint: disable=protected-access,too-many-arguments,too-many-locals

def set_spec2vec_defaults(**settings):
    """Set spec2vec default argument values"(where no user input is given)".

    Args
    ----------
    **settings: optional dict
        Change default settings
    """
    defaults = {"mz_from": 0,
                "mz_to": 1000,
                "n_required": 10,
                "ratio_desired": 0.5,
                "intensity_from": 0.001,
                "loss_mz_from": 5.0,
                "loss_mz_to": 200.0,
                "intensity_weighting_power": 0.5,
                "allowed_missing_percentage": 0,
                "cosine_tol": 0.005,
                "mass_tolerance": 1.0,
                "ignore_non_annotated": True}

    # Set default parameters or replace by **settings input
    for key in defaults:
        if key not in settings:
            settings[key] = defaults[key]
    return settings


def post_process_s2v(spectrum, **settings):
    """Returns a processed matchms.Spectrum.Spectrum

    Args:
    ----------
    spectrum: matchms.Spectrum.Spectrum
        Spectrum to process
    mz_from
        Set lower threshold for m/z peak positions. Default is 0.0.
    mz_to
        Set upper threshold for m/z peak positions. Default is 1000.0.
    n_required
        Number of minimal required peaks for a spectrum to be considered.
    ratio_desired
        Number of minimum required peaks. Spectra with fewer peaks will be set
        to 'None'. Default is 1.
    intensity_from
        Set lower threshold for peak intensity. Default is 10.0.
    loss_mz_from
        Minimum allowed m/z value for losses. Default is 0.0.
    loss_mz_to
        Maximum allowed m/z value for losses. Default is 1000.0.
    """
    settings = set_spec2vec_defaults(**settings)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=settings["mz_from"],
                            mz_to=settings["mz_to"])
    spectrum = require_minimum_number_of_peaks(spectrum,
                                               n_required=settings[
                                                   "n_required"])
    spectrum = reduce_to_number_of_peaks(spectrum,
                                         n_required=settings["n_required"],
                                         ratio_desired=settings
                                         ["ratio_desired"])
    if spectrum is None:
        return None
    s_remove_low_peaks = select_by_relative_intensity(spectrum,
                                                      intensity_from=settings
                                                      ["intensity_from"])
    if len(s_remove_low_peaks.peaks) >= 10:
        spectrum = s_remove_low_peaks

    spectrum = add_losses(spectrum, loss_mz_from=settings["loss_mz_from"],
                          loss_mz_to=settings["loss_mz_to"])
    return spectrum


def process_spectrums(spectrums, **settings):
    """Returns list of post-processed SpectrumDocuments from input spectrums

    Args:
    ----------
    spectrums: list of matchms.Spectrum.Spectrum
        Input spectra
    mz_from
        Set lower threshold for m/z peak positions. Default is 0.0.
    mz_to
        Set upper threshold for m/z peak positions. Default is 1000.0.
    n_required
        Number of minimal required peaks for a spectrum to be considered.
    ratio_desired
        Number of minimum required peaks. Spectra with fewer peaks will be set
        to 'None'. Default is 1.
    intensity_from
        Set lower threshold for peak intensity. Default is 10.0.
    loss_mz_from
        Minimum allowed m/z value for losses. Default is 0.0.
    loss_mz_to
        Maximum allowed m/z value for losses. Default is 1000.0.
    """
    spectrums = [post_process_s2v(spec, **settings) for spec in spectrums]
    documents = [SpectrumDocument(spec, n_decimals=2) for spec in spectrums]

    return documents


def get_metadata(documents: List[SpectrumDocument]):
    """Returns a list of smiles (str) from the input spectrum documents

    Args:
    --------
    documents: list of SpectrumDocument
    """
    metadata = []
    for doc in documents:
        metadata.append(doc._obj.get("smiles"))
    return metadata


def search_topn_s2v_matches(documents_query: List[SpectrumDocument],
                            documents_library: List[SpectrumDocument],
                            model: BaseTopicModel,
                            library_ids: Union[List[int], np.ndarray],
                            presearch_based_on: List[str] = ("parentmass",
                                                             "spec2vec-top10"),
                            intensity_weighting_power: float = 0.5,
                            allowed_missing_percentage: float = 0):
    """
    Returns (ndarray, ndarray) recording topn library IDs, s2v scores per query

    First ndarray records topn library IDs for each query. It has
    shape(topn, len(queries)). The second ndarray are the s2v scores against
    all library documents per query. It has shape (len(library), len(queries)).

    Args:
    -------
    documents_query:
        List containing all spectrum documents that should be queried against
        the library.
    documents_library:
        List containing all library spectrum documents.
    model:
        Pretrained word2Vec model.
    library_ids: list-like of int
        List with library ids to consider for spec2vec matching.
    presearch_based_on: list, optional
        What to select candidates on. Options are now: parentmass,
        spec2vec-topX where X can be any number. Default = ("parentmass",
        "spec2vec-top10")
    intensity_weighting_power: float, optional
        Spectrum vectors are a weighted sum of the word vectors. The given word
        intensities will be raised to the given power. Default = 0.5.
    allowed_missing_percentage: float, optional
        Set the maximum allowed percentage of the document that may be missing
        from the input model. This is measured as percentage of the weighted,
        missing words compared to all word vectors of the document. Default = 0
        which means no missing words are allowed.
    """
    m_spec2vec_similarities = None

    if np.any(["spec2vec" in x for x in presearch_based_on]):
        top_n = int(
            [x.split("top")[1] for x in presearch_based_on if "spec2vec" in x][
                0])
        print("Pre-selection includes spec2vec top {}.".format(top_n))
        spec2vec = Spec2Vec(
            model=model, intensity_weighting_power=intensity_weighting_power,
            allowed_missing_percentage=allowed_missing_percentage)
        m_spec2vec_similarities = spec2vec.matrix(
            [documents_library[i] for i in library_ids],
            documents_query)

        # Select top_n similarity values:
        selection_spec2vec = np.argpartition(m_spec2vec_similarities, -top_n,
                                             axis=0)[-top_n:, :]
    else:
        selection_spec2vec = np.empty((0, len(documents_query)), dtype="int")

    return selection_spec2vec, m_spec2vec_similarities


def search_parent_mass_matches(documents_query: List[SpectrumDocument],
                               documents_library: List[SpectrumDocument],
                               library_ids: Union[List[int], np.ndarray],
                               presearch_based_on: List[str] = (
                                       "parentmass", "spec2vec-top10"),
                               mass_tolerance: float = 1.0):
    """
    Returns (list, ndarray) of parent mass matching library IDs, s2v scores

    First list records all parent mass matching library IDs for each query.
    The second ndarray are the mass match scores against all library documents
    per query. It has shape (len(library), len(queries)).

    Args:
    -------
    documents_query:
        List containing all spectrum documents that should be queried against
        the library.
    documents_library:
        List containing all library spectrum documents.
    model:
        Pretrained word2Vec model.
    library_ids: list-like of int
        List with library ids to consider.
    presearch_based_on: list, optional
        What to select candidates on. Options are now: parentmass,
        spec2vec-topX where X can be any number. Default = ("parentmass",
        "spec2vec-top10")
    mass_tolerance: float, optional
        Specify tolerance for a parentmass match. Default = 1.
    """
    m_mass_matches = None

    if "parentmass" in presearch_based_on:
        mass_matching = ParentMassMatch(mass_tolerance)
        m_mass_matches = mass_matching.matrix(
            [documents_library[i]._obj for i in library_ids],
            [x._obj for x in documents_query])
        selection_massmatch = []
        for i in range(len(documents_query)):
            selection_massmatch.append(np.where(m_mass_matches[:, i] == 1)[0])
    else:
        selection_massmatch = np.empty((len(documents_query), 0), dtype="int")

    return selection_massmatch, m_mass_matches


def find_matches(document_query: SpectrumDocument,
                 documents_library: List[SpectrumDocument],
                 model: BaseTopicModel,
                 library_ids: Union[List[int], np.ndarray],
                 all_match_ids: Union[List[int], np.ndarray],
                 i: int,
                 m_spec2vec_similarities: Union[np.ndarray, float],
                 m_mass_matches: Union[np.ndarray, float],
                 include_scores: List[str] = ("spec2vec", "cosine",
                                              "modcosine"),
                 intensity_weighting_power: float = 0.5,
                 allowed_missing_percentage: float = 0,
                 cosine_tol: float = 0.005):
    """Finds library matches for one query document, returns pd.DataFrame

    Args:
    -------
    document_query:
        Spectrum documents that should be queried against
        the library.
    documents_library:
        List containing all library spectrum documents.
    model:
        Pretrained word2Vec model.
    library_ids: list-like of int
        List with library ids to consider for spec2vec matching.
    all_match_ids: list-like of int
        List with all library ids to consider (also parentmass match).
    i:
        Keeps track of current query number
    selection_spec2vec:
        The topn library IDs for each query. It has shape(topn, len(queries)).
    m_spec2vec_similarities: ndarray of float
        The second ndarray are the s2v scores against all library documents per
        query. It has shape (len(library), len(queries)).
    m_mass_matches: ndarray of float
        The mass match scores against all library documents per query. It has
        shape (len(library), len(queries)).
    include_scores:
        Scores to include in output. Default = ("spec2vec", "cosine",
        "modcosine")
    intensity_weighting_power:
        Spectrum vectors are a weighted sum of the word vectors. The given word
        intensities will be raised to the given power. Default = 0.5.
    allowed_missing_percentage:
        Set the maximum allowed percentage of the document that may be missing
        from the input model. This is measured as percentage of the weighted,
        missing words compared to all word vectors of the document. Default = 0
        which means no missing words are allowed.
    cosine_tol:
        Set tolerance for the cosine and modified cosine score. Default = 0.005
    """
    if "cosine" in include_scores:
        # Get cosine score for found matches
        cosine_similarity = CosineGreedy(tolerance=cosine_tol)
        cosine_scores = []
        for match_id in library_ids[all_match_ids]:
            cosine_scores.append(cosine_similarity.matrix(
                [documents_library[match_id]._obj],
                [document_query._obj]))
    else:
        cosine_scores = len(all_match_ids) * ["not calculated"]

    if "modcosine" in include_scores:
        # Get modified cosine score for found matches
        mod_cosine_similarity = ModifiedCosine(tolerance=cosine_tol)
        mod_cosine_scores = []
        for match_id in library_ids[all_match_ids]:
            mod_cosine_scores.append(mod_cosine_similarity.matrix(
                [documents_library[match_id]._obj],
                [document_query._obj]))
    else:
        mod_cosine_scores = len(all_match_ids) * ["not calculated"]

    matches_df = pd.DataFrame(
        {"cosine_score": [x[0, 0][0] for x in cosine_scores],
         "cosine_matches": [x[0, 0][1] for x in cosine_scores],
         "mod_cosine_score": [x[0, 0][0] for x in mod_cosine_scores],
         "mod_cosine_matches": [x[0, 0][1]
                                for x in mod_cosine_scores]},
        index=library_ids[all_match_ids])

    if m_mass_matches is not None:
        matches_df["mass_match"] = m_mass_matches[all_match_ids, i]

    if m_spec2vec_similarities is not None:
        matches_df["s2v_score"] = m_spec2vec_similarities[
            all_match_ids, i]
    elif "spec2vec" in include_scores:
        spec2vec_similarity = Spec2Vec(
            model=model,
            intensity_weighting_power=intensity_weighting_power,
            allowed_missing_percentage=allowed_missing_percentage)
        spec2vec_scores = []
        for match_id in library_ids[all_match_ids]:
            spec2vec_scores.append(
                spec2vec_similarity.pair(documents_library[match_id],
                                         document_query))
        matches_df["s2v_score"] = spec2vec_scores

    return matches_df


def combine_found_matches(documents_query: List[SpectrumDocument],
                          documents_library: List[SpectrumDocument],
                          model: BaseTopicModel,
                          library_ids: Union[List[int], np.ndarray],
                          selection_spec2vec,
                          m_spec2vec_similarities: Union[np.ndarray, float],
                          selection_massmatch: List[int],
                          m_mass_matches: Union[np.ndarray, float],
                          include_scores: List[str] = ("spec2vec", "cosine",
                                                       "modcosine"),
                          intensity_weighting_power: float = 0.5,
                          allowed_missing_percentage: float = 0,
                          cosine_tol: float = 0.005):
    """
    Finds library matches for all query documents, returns list of pd.DataFrame

    Args:
    --------
    documents_query:
        List containing all spectrum documents that should be queried against
        the library.
    documents_library:
        List containing all library spectrum documents.
    model:
        Pretrained word2Vec model.
    library_ids: list-like of int
        List with library ids to consider for spec2vec matching.
    selection_spec2vec: ndarray of int
        The topn library IDs for each query. It has shape(topn, len(queries)).
    m_spec2vec_similarities:
        The second ndarray are the s2v scores against all library documents per
        query. It has shape (len(library), len(queries)).
    selection_massmatch:
        List of all parent mass matching library IDs for each query.
    m_mass_matches:
        The mass match scores against all library documents per query. It has
        shape (len(library), len(queries)).
    include_scores: list, optional
        Scores to include in output. Default = ("spec2vec", "cosine",
        "modcosine")
    intensity_weighting_power: float, optional
        Spectrum vectors are a weighted sum of the word vectors. The given word
        intensities will be raised to the given power. Default = 0.5.
    allowed_missing_percentage: float, optional
        Set the maximum allowed percentage of the document that may be missing
        from the input model. This is measured as percentage of the weighted,
        missing words compared to all word vectors of the document. Default = 0
        which means no missing words are allowed.
    cosine_tol: float, optional
        Set tolerance for the cosine and modified cosine score. Default = 0.005
    """
    combined_matches = []

    for i, document_query in enumerate(documents_query):
        s2v_top_ids = selection_spec2vec[:, i]
        mass_match_ids = selection_massmatch[i]

        all_match_ids = np.unique(
            np.concatenate((s2v_top_ids, mass_match_ids)))

        if len(all_match_ids) > 0:
            matches_df = find_matches(document_query, documents_library,
                                      model, library_ids, all_match_ids,
                                      i, m_spec2vec_similarities,
                                      m_mass_matches, include_scores,
                                      intensity_weighting_power,
                                      allowed_missing_percentage,
                                      cosine_tol)
            combined_matches.append(matches_df.fillna(0))
        else:
            combined_matches.append([])

    return combined_matches


def library_matching(documents_query: List[SpectrumDocument],
                     documents_library: List[SpectrumDocument],
                     model: BaseTopicModel,
                     presearch_based_on: List[str] = ("parentmass",
                                                      "spec2vec-top10"),
                     include_scores: List[str] = ("spec2vec", "cosine",
                                                  "modcosine"),
                     **settings):
    """Selecting potential spectra matches with spectra library.

    Suitable candidates will be selected by 1) top_n Spec2Vec similarity, and
    2) same precursor mass (within given mz_ppm tolerance(s)).
    For later matching routines, additional scores (cosine, modified cosine)
    are added as well.

    Args:
    --------
    documents_query:
        List containing all spectrum documents that should be queried against
        the library.
    documents_library:
        List containing all library spectrum documents.
    model:
        Pretrained word2Vec model.
    presearch_based_on: list, optional
        What to select candidates on. Options are now: parentmass,
        spec2vec-topX where X can be any number. Default = ("parentmass",
        "spec2vec-top10")
    include_scores: list, optional
        Scores to include in output. Default = ("spec2vec", "cosine",
        "modcosine")
    ignore_non_annotated: bool, optional
        If True, only annotated spectra will be considered for matching.
        Default = True.
    intensity_weighting_power: float, optional
        Spectrum vectors are a weighted sum of the word vectors. The given word
        intensities will be raised to the given power. Default = 0.5.
    allowed_missing_percentage: float, optional
        Set the maximum allowed percentage of the document that may be missing
        from the input model. This is measured as percentage of the weighted,
        missing words compared to all word vectors of the document. Default = 0
        which means no missing words are allowed.
    cosine_tol: float, optional
        Set tolerance for the cosine and modified cosine score. Default = 0.005
    mass_tolerance: float, optional
        Specify tolerance for a parentmass match. Default = 1.
    """

    # Initialise, error message
    settings = set_spec2vec_defaults(**settings)
    library_spectra_metadata = get_metadata(documents_library)
    if settings["ignore_non_annotated"]:
        # Get array of all ids for spectra with smiles
        library_ids = np.asarray([i for i, x in enumerate(
            library_spectra_metadata) if x])
    else:
        library_ids = np.arange(len(documents_library))

    msg = "Presearch must be done either by 'parentmass' and/or" + \
          "'spec2vec-topX'"
    assert "parentmass" in presearch_based_on or \
           np.any(["spec2vec" in x for x in presearch_based_on]), msg

    # 1. Search for top-n Spec2Vec matches ------------------------------------
    selection_spec2vec, m_spec2vec_similarities = search_topn_s2v_matches(
        documents_query, documents_library, model, library_ids,
        presearch_based_on, settings["intensity_weighting_power"],
        settings["allowed_missing_percentage"])

    # 2. Search for parent mass based matches ---------------------------------
    selection_massmatch, m_mass_matches = search_parent_mass_matches(
        documents_query, documents_library, library_ids, presearch_based_on,
        settings["mass_tolerance"])

    # 3. Combine found matches ------------------------------------------------
    found_matches = combine_found_matches(
        documents_query, documents_library, model, library_ids,
        selection_spec2vec, m_spec2vec_similarities, selection_massmatch,
        m_mass_matches, include_scores, settings["intensity_weighting_power"],
        settings["allowed_missing_percentage"], settings["cosine_tol"])
    return found_matches
