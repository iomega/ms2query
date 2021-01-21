"""
Functions to create and retrieve information from sqlite files
"""

import sqlite3
import json
from typing import Dict, List
import ast
from matchms.importing.load_from_json import dict2spectrum
from matchms.Spectrum import Spectrum
from pandas.io.sql import DataFrame
import pandas as pd
import numpy as np
import os
import time


def make_sqlfile_wrapper(sqlite_file_name: str,
                         columns_dict: Dict[str, str],
                         json_spectrum_file_name: str
                         ):
    """Creates a new sqlite file, with columns defined in columns_dict

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists the
        table spectra is overwritten.
    columns_dict:
        Dictionary with as keys the column names and as values the sql datatype
    json_spectrum_file_name:
        File location of the json file that stores the spectra.
    table_name:
        Name of the table that is created in the sqlite file,
        default = "spectra"
    """
    create_table_structure(sqlite_file_name, columns_dict)
    add_spectra_to_database(sqlite_file_name, json_spectrum_file_name)

def create_table_structure(sqlite_file_name: str,
                           columns_dict: Dict[str, str],
                           table_name: str = "spectra"):
    """Creates a new sqlite file, with columns defined in columns_dict

    Args:
    -------
    sqlite_file_name:
        Name of sqlite_file that should be created, if it already exists the
        table spectra is overwritten.
    columns_dict:
        Dictionary with as keys the column names and as values the sql datatype
    table_name:
        Name of the table that is created in the sqlite file,
        default = "spectra"
    """
    create_table_command = f"""
    DROP TABLE IF EXISTS {table_name};
    CREATE TABLE {table_name} (
    """
    # add all columns with the type specified in columns_dict
    for column_header in columns_dict:
        create_table_command += column_header + " " \
                                + columns_dict[column_header] + ",\n"
    create_table_command += "full_json VARCHAR,\n"
    create_table_command += "PRIMARY KEY (spectrum_id));"

    conn = sqlite3.connect(sqlite_file_name)
    cur = conn.cursor()
    cur.executescript(create_table_command)
    conn.commit()
    conn.close()


def add_spectra_to_database(sqlite_file_name: str,
                            json_spectrum_file_name: str,
                            table_name: str = "spectra"):
    """Creates a sqlite file containing the information from a json file.

     Args:
    -------
    sqlite_file_name:
        Name of sqlite file to which the spectra should be added.
    json_spectrum_file_name:
        File location of the json file that stores the spectra.
    table_name:
        Name of the table in the database to which the spectra should be added,
        default = "spectra"
    """
    spectra = json.load(open(json_spectrum_file_name))
    conn = sqlite3.connect(sqlite_file_name)

    # Get the column names in the sqlite file
    cur = conn.execute(f'select * from {table_name}')
    column_names = list(map(lambda x: x[0], cur.description))

    # Add the information of each spectrum to the sqlite file
    for spectrum in spectra:
        columns = ""
        values = ""
        # Check if there is a key for the spectrum that is the same as the
        # specified columns and if this is the case add this value to this
        # column.
        for column in column_names:
            if column in spectrum:
                # Add comma when it is not the first column that is added
                if len(columns) > 0:
                    columns += ", "
                    values += ", "
                columns += column
                values += '"' + spectrum[column] + '"'

            # Add the complete spectrum in json format to the column full_json
            elif column == "full_json":
                if len(columns) > 0:
                    columns += ", "
                    values += ", "
                columns += "full_json"
                values += f'"{spectrum}"'
            else:
                print(f"No value found for column: {column} in "
                      f"spectrum {spectrum['spectrum_id']}")
        add_spectrum_command = f"INSERT INTO {table_name} " \
                               + f"({columns}) values ({values})"

        cur = conn.cursor()
        cur.execute(add_spectrum_command)
        conn.commit()
    conn.close()


def get_spectra_from_sqlite(sqlite_file_name: str,
                            spectrum_id_list: List[str],
                            table_name: str = "spectra") -> List[Spectrum]:
    """Returns a list with all metadata of spectrum_ids in spectrum_id_list

    Args:
    -------
    sqlite_file_name:
        File name of the sqlite file that contains the spectrum information
    spectrum_id_list:
        List of spectrum_id's of which the metadata should be returned
    table_name:
        Name of the table in the sqlite file that stores the spectrum data
    """
    conn = sqlite3.connect(sqlite_file_name)

    sqlite_command = f"""SELECT full_json FROM {table_name} 
                    WHERE spectrum_id 
                    IN ('{"', '".join(map(str, spectrum_id_list))}')"""

    cur = conn.cursor()
    cur.execute(sqlite_command)

    list_of_spectra_dict = []
    for json_spectrum in cur:
        # Remove the "()" around the spectrum
        json_spectrum = json_spectrum[0]
        # Convert string to dictionary
        json_spectrum = ast.literal_eval(json_spectrum)
        list_of_spectra_dict.append(json_spectrum)
    conn.close()

    # Convert to matchms.Spectrum.Spectrum object
    spectra_list = []
    for spectrum in list_of_spectra_dict:
        spectra_list.append(dict2spectrum(spectrum))
    return spectra_list


def convert_dataframe_to_sqlite(spectrum_dataframe: pd.DataFrame,
                                file_name: str):
    connection = sqlite3.connect(file_name)

    # Optimal chuncksize: 20000 = 16.5 s, 30000 = 16,3 s
    spectrum_dataframe.to_sql('tanimoto', connection,
                              method='multi',
                              chunksize=30000,
                              if_exists="append",
                              index=False)
    connection.commit()
    connection.close()


def get_inchikey_order(metadata_file: str =
                          "../downloads/metadata_AllInchikeys14.csv"
                          ) -> List[str]:
    """Return list of Inchi14s in same order as in metadata_file

    Args:
    ------
    metadata_file:
        path to metadata file, expected format is csv, with inchikeys in the
        second column, starting from the second row. Default =
        "../downloads/metadata_AllInchikeys14.csv"
    """
    with open(metadata_file, 'r') as inf:
        inf.readline()
        inchi_list = []
        for line in inf:
            line = line.strip().split(',')
            inchi_list.append(line[1])
    return inchi_list


def create_sqlite_table_for_tanimoto_scores_without_duplicates(sqlite_file_name: str):
    create_table_command = """;
    DROP TABLE IF EXISTS tanimoto;
    CREATE TABLE tanimoto (
        inchikey1 INTEGER,
        inchikey2 INTEGER,
        tanimoto_score REAL,
        PRIMARY KEY (inchikey1, inchikey2)
        );
    """

    conn = sqlite3.connect(sqlite_file_name)
    cur = conn.cursor()
    cur.executescript(create_table_command)
    conn.commit()
    conn.close()


    tanimoto_score_matrix = np.load(
        "../downloads/similarities_AllInchikeys14_daylight2048_jaccard.npy", mmap_mode='r')
    list_of_numbers = []
    for i in range(len(tanimoto_score_matrix[0])):
        list_of_numbers.append(i)

    start_time = time.time()
    for row_nr, row in enumerate(tanimoto_score_matrix):
        # Remove duplicates
        row = row[:row_nr+1]
        df = pd.DataFrame(row)

        # Add
        df['inchikey1'] = np.array(len(row) * [row_nr])
        df['inchikey2'] = np.array(list_of_numbers[:len(row)])
        df.rename(columns={0 :'tanimoto_score'}, inplace=True)
        # add table with dataframe to sqlite file
        convert_dataframe_to_sqlite(df, sqlite_file_name)
        if row_nr % 100 == 0:
            print(row_nr)
    print("--- %s seconds ---" % (time.time() - start_time))



def create_sqlite_table_for_tanimoto_scores(sqlite_file_name: str):
    create_table_command = """;
    DROP TABLE IF EXISTS tanimoto;
    CREATE TABLE tanimoto (
        inchikey1 INTEGER,
        inchikey2 INTEGER,
        tanimoto_score REAL,
        PRIMARY KEY (inchikey1, inchikey2)
        );
    """

    conn = sqlite3.connect(sqlite_file_name)
    cur = conn.cursor()
    cur.executescript(create_table_command)
    conn.commit()
    conn.close()

    #Running for matrix of length 12000 takes about 4 minutes
    inchikey_order = get_inchikey_order()
    list_of_numbers = []
    for i in range(len(inchikey_order)):
        list_of_numbers.append(i)
    inchikey_order = list_of_numbers
    tanimoto_score_matrix = np.load(
        "../downloads/similarities_AllInchikeys14_daylight2048_jaccard.npy", mmap_mode='r')
    # for row in tanimoto_score_matrix


    start_time = time.time()
    for i in range(0, len(inchikey_order), 20000):
        part_of_tanimoto_score_matrix = tanimoto_score_matrix[i:i+200]
        tanimoto_score_matrix = ""
        print('1')
        df = pd.DataFrame(part_of_tanimoto_score_matrix,
                          columns=inchikey_order)
        part_of_tanimoto_score_matrix = ""
        print('2')
        df['inchikey1'] = inchikey_order[i:i+200]
        print('3')
        df = df.melt(id_vars='inchikey1', var_name='inchikey2', value_name='tanimoto_score', ignore_index=True)
        print('4')
        # df['inchikey2'] = df['inchikey2'].astype('int64')
        print('5')
        # df.set_index(['inchikey1', 'inchikey2'], inplace=True)

        # df.drop_duplicates(subset=['inchikey1', ])
        # add table with dataframe to sqlite file
        convert_dataframe_to_sqlite(df, sqlite_file_name)
        print("--- %s seconds ---" % (time.time() - start_time))
    return df


    # new_np_array = np.array([["","",""]])
    # for index1, tanimoto_score_array in enumerate(tanimoto_score_matrix):
    #     for index2 in range(len(tanimoto_score_array)):
    #         to_add = np.array([[inchikey_order[index1], inchikey_order[index2], tanimoto_score_matrix[index1][index2]]])
    #         new_np_array = np.append(new_np_array, to_add, axis=0)
    #     print(index1)


def get_tanimoto_from_sqlite(sqlite_file_name: str):
    start_time = time.time()

    conn = sqlite3.connect(sqlite_file_name)

    sqlite_command = """SELECT inchikey1, inchikey2, tanimoto_score FROM tanimoto 
                    WHERE inchikey1 in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,200) and inchikey2 in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,200);
                    """

    cur = conn.cursor()
    cur.execute(sqlite_command)
    for score in cur:
        print(score)
    conn.close()
    print("--- %s seconds ---" % (time.time() - start_time))

    # return tanimoto_score

def make_inchikeys_composite_key(sqlite_file_name: str):
    create_table_command = """ALTER TABLE tanimoto 
    ADD PRIMARY KEY(inchikey1, inchikey2);
    
    """

    conn = sqlite3.connect(sqlite_file_name)
    cur = conn.cursor()
    cur.execute(create_table_command)
    conn.commit()
    conn.close()

def remove_duplicates_from_csv(tanimoto_score_file_location, new_file_name):
    tanimoto_scores = np.load(tanimoto_score_file_location, mmap_mode='r')
    start_time = time.time()

    with open(new_file_name + ".txt", 'w') as txt_file:
        for row_nr, row in enumerate(tanimoto_scores):
            kept_part = row[:row_nr+1]
            new_row = np.concatenate((kept_part, np.array([None]*(len(row)-row_nr-1))))
            for entry in new_row:
                txt_file.write(str(entry) + ",")
            if row_nr % 100 == 0:
                print(row_nr)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    with open(new_file_name + ".txt", 'r') as txt_file:
        no_duplicate_scores = txt_file.read()
    no_duplicate_scores = no_duplicate_scores.split(',')
    no_duplicate_scores.pop()
    no_duplicate_scores = np.array(no_duplicate_scores).reshape(len(row), len(row))
    np.save(new_file_name + ".npy", no_duplicate_scores)
    print(no_duplicate_scores)


if __name__ == "__main__":
    # spectrum_dataframe = pd.DataFrame({'name': ['User 1', 'User 2', 'User 3']})
    # convert_dataframe_to_sqlite(spectrum_dataframe)
    # data = np.load("../downloads/similarities_AllInchikeys14_daylight2048_jaccard.npy")
    # my_array = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
    # remove_duplicates_from_csv("../downloads/similarities_AllInchikeys14_daylight2048_jaccard.npy", "tanimoto_without_duplicates")
    # create_sqlite_table_for_tanimoto_scores_without_duplicates("dataframe_test_file.sqlite")
    # print(df)
    # print(len(df))
    # print(df.dtypes)
    # print(df.memory_usage(deep= True))
    # make_inchikeys_composite_key("dataframe_test_file_without_index.sqlite")

    get_tanimoto_from_sqlite("dataframe_test_file.sqlite")
    """
    PERFORMANCE SCORES:
    run 4000 rows in 40 minutes, when created with two primary keys
    run 12000 rows in 10 minutes, when created with integers instead of inchikeys"""