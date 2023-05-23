import json
import urllib
from http.client import InvalidURL
from typing import List, Optional

import pandas as pd
from tqdm import tqdm


def select_inchikeys(spectra):
    list_of_inchikeys = []
    for spectrum in spectra:
        list_of_inchikeys.append(spectrum.get("inchikey")[:14])
    return set(list_of_inchikeys)


def select_smiles_and_full_inchikeys(spectra):
    """"""
    set_of_inchikey14s = select_inchikeys(spectra)
    inchikey_dict = {inchikey: [] for inchikey in set_of_inchikey14s}
    for spectrum in spectra:
        inchikey = spectrum.get("inchikey")
        smiles = spectrum.get("smiles")
        inchikey_dict[inchikey[:14]].append([inchikey, smiles])
    return inchikey_dict


def do_url_request(url: str) -> [bytes, None]:
    """
    Do url request and return bytes from .read() or None if HTTPError is raised
    :param url: url to access
    :return: open file or None if request failed
    """
    try:
        with urllib.request.urlopen(url) as inf:
            result = inf.read()
    except (urllib.error.HTTPError, urllib.error.URLError, InvalidURL):
        # apparently the request failed
        result = None
    except ConnectionAbortedError:
        result = do_url_request(url)
        print("An connection error occurred, will try again")
    return result


def get_json_cf_results(full_inchikey: str) -> Optional[List[str]]:
    """
    Extract the wanted CF classes.
    Names of the keys extracted in order are:
    'kingdom', 'superclass', 'class', 'subclass', 'direct_parent'
    """
    raw_json = do_url_request(f"http://classyfire.wishartlab.com/entities/{full_inchikey}.json")
    if raw_json is None:
        return None
    classes = []
    cf_json = json.loads(raw_json)
    wanted_keys_list_name = ['kingdom', 'superclass', 'class',
                             'subclass', 'direct_parent']
    for key in wanted_keys_list_name:
        info_dict = cf_json.get(key, "")
        info = ""
        if info_dict:
            info = info_dict.get('name', "")
        classes.append(info)
    assert len(classes) == len(wanted_keys_list_name), "expected a different number of metadata"
    return classes


def get_json_npc_results(smiles: str) -> Optional[List[str]]:
    """Read bytes version of json str, extract the keys in order
    Names of the keys extracted in order are:
    class_results, superclass_results, pathway_results, isglycoside.
    List elements are concatonated with '; '.
    :param raw_json:Json str as a bytes object containing NPClassifier
        information
    :return: Extracted NPClassifier classes
    """
    raw_json = do_url_request(f"https://npclassifier.ucsd.edu/classify?smiles={smiles}")
    if raw_json is None:
        return None
    wanted_info = []
    cf_json = json.loads(raw_json)
    wanted_keys_list = ["class_results", "superclass_results",
                        "pathway_results"]
    # this one returns a bool not a list like the others
    last_key = "isglycoside"

    for key in wanted_keys_list:
        info_list = cf_json.get(key, "")
        info = ""
        if info_list:
            info = "; ".join(info_list)
        wanted_info.append(info)

    last_info_bool = cf_json.get(last_key, "")
    last_info = "False"
    if last_info_bool:
        last_info = "True"
    wanted_info.append(last_info)
    assert len(wanted_info) == len(wanted_keys_list)+1, "expected a different number of metadata"
    return wanted_info


def select_compound_classes(spectra):
    inchikey_dict = select_smiles_and_full_inchikeys(spectra)
    inchikey_results_list = []
    for i, inchikey14 in tqdm(enumerate(inchikey_dict), total=len(inchikey_dict),
                              desc="Adding compound class annotation to inchikeys"):
        inchikey_results_list.append([inchikey14])
        # select classyfire classes
        cf_classes = None
        for full_inchikey, smiles in inchikey_dict[inchikey14]:
            cf_classes = get_json_cf_results(full_inchikey)
            if cf_classes is not None:
                inchikey_results_list[i] += cf_classes
                break
        if cf_classes is None:
            print(f"no classyfire annotation was found for inchikey {inchikey14}")
            inchikey_results_list[i].append([inchikey14, "", "", "", "", ""])

    #     select NPC classes
        npc_results = None
        for _, smiles in inchikey_dict[inchikey14]:
            npc_results = get_json_npc_results(smiles)
            if npc_results is not None:
                inchikey_results_list[i] += npc_results
                break
        if npc_results is None:
            print(f"no npc annotation was found for inchikey {inchikey14}")
            inchikey_results_list[i] += ["", "", "", ""]
    return inchikey_results_list


def convert_to_dataframe(inchikey_results_lists)->pd.DataFrame:
    header_list = [
        'inchikey', 'cf_kingdom',
        'cf_superclass', 'cf_class', 'cf_subclass', 'cf_direct_parent',
        'npc_class_results', 'npc_superclass_results', 'npc_pathway_results',
        'npc_isglycoside']
    df = pd.DataFrame(inchikey_results_lists, columns=header_list)
    df.set_index("inchikey", inplace=True)
    return df
