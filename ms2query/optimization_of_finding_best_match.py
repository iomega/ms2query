import sqlite3
import pickle
from ms2query.app_helpers import load_pickled_file
from ms2query.query_from_sqlite_database import get_index_of_inchikeys, \
    get_spectra_from_sqlite


def get_index_of_inchikey_in_sqlite(sqlite_file):
    all_spectra = get_spectra_from_sqlite(sqlite_file, [],
                                          get_all_spectra=True)
    inchikeys = {}
    for spectrum in all_spectra:
        inchikey = spectrum.get("inchikey")[:14]
        inchikeys[inchikey] = True
    inchikeys = list(inchikeys.keys())
    indexes = get_index_of_inchikeys(inchikeys, sqlite_file)
    pickle.dump(indexes, open("../downloads/optimizing_preselection/indexes_of_inchikeys_in_training_spectra", "wb"))


def find_highest_tanimoto_score(sqlite_file, query_spectrum,
                                available_inchikeys_file: str):
    inchikey14 = query_spectrum.get("inchikey")[:14]
    assert len(inchikey14) == 14, "Expected an inchikey of length 14"
    index = get_index_of_inchikeys([inchikey14], sqlite_file)[inchikey14]

    indexes_of_available_inchikeys = load_pickled_file(available_inchikeys_file)
    indexes = list(indexes_of_available_inchikeys.values())
    conn = sqlite3.connect(sqlite_file)

    sqlite_command = f"""SELECT identifier_2, tanimoto_score
                        FROM tanimoto_scores
                        WHERE identifier_1 = "{index}"
                        UNION
                        SELECT identifier_1, tanimoto_score
                        FROM tanimoto_scores
                        WHERE identifier_2 = "{index}"
                        ;
                        """
    cur = conn.cursor()
    cur.execute(sqlite_command)
    results = cur.fetchall()
    tanimoto_scores = {}
    for result in results:
        index = result[0]
        tanimoto_score = result[1]
        if index in indexes:
            tanimoto_scores[index] = tanimoto_score
    highest_tanimoto_score = max(tanimoto_scores, key=tanimoto_scores.get)
    print(highest_tanimoto_score)


if __name__ == "__main__":
    test_spectrum = load_pickled_file(
        "../tests/test_files/general_test_files/100_test_spectra.pickle")[0]
    sqlite_file = "../downloads/train_ms2query_nn_data/" \
                  "ALL_GNPS_positive_train_split_210305.sqlite"
    # sqlite_file = "../tests/test_files/general_test_files/100_test_spectra.sqlite"
    find_highest_tanimoto_score(
        sqlite_file, test_spectrum,
        "../downloads/optimizing_preselection/indexes_of_inchikeys_in_training_spectra")
    # get_index_of_inchikey_in_sqlite(sqlite_file)

