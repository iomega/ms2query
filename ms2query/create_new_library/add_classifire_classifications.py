import json
import urllib
from typing import List, Union, Dict
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
    except (urllib.error.HTTPError, urllib.error.URLError):
        # apparently the request failed
        result = None
    return result


def get_json_cf_results(raw_json: bytes) -> List[str]:
    """
    Extract the wanted CF classes from bytes version (open file) of json str
    Names of the keys extracted in order are:
    'kingdom', 'superclass', 'class', 'subclass', 'direct_parent'
    List elements are concatonated with '; '.
    :param raw_json: Json str as a bytes object containing ClassyFire
        information
    :return: Extracted CF classes
    """
    wanted_info = []
    cf_json = json.loads(raw_json)
    wanted_keys_list_name = ['kingdom', 'superclass', 'class',
                             'subclass', 'direct_parent']
    for key in wanted_keys_list_name:
        info_dict = cf_json.get(key, "")
        info = ""
        if info_dict:
            info = info_dict.get('name', "")
        wanted_info.append(info)

    return wanted_info


def get_json_npc_results(raw_json: bytes) -> List[str]:
    """Read bytes version of json str, extract the keys in order
    Names of the keys extracted in order are:
    class_results, superclass_results, pathway_results, isglycoside.
    List elements are concatonated with '; '.
    :param raw_json:Json str as a bytes object containing NPClassifier
        information
    :return: Extracted NPClassifier classes
    """
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
    last_info = "0"
    if last_info_bool:
        last_info = "1"
    wanted_info.append(last_info)

    return wanted_info


def select_compound_classes(inchikey_dict):
    inchikey_results_list = []
    for i, inchikey14 in tqdm(enumerate(inchikey_dict), total=len(inchikey_dict)):
        inchikey_results_list.append([inchikey14])
        # select classyfire classes
        for full_inchikey, smiles in inchikey_dict[inchikey14]:
            result = do_url_request(f"http://classyfire.wishartlab.com/entities/{full_inchikey}.json")
            if result is not None:
                classes = get_json_cf_results(result)
                inchikey_results_list[i].append(smiles)
                inchikey_results_list[i] += classes
                break
        if len(inchikey_results_list[i]) != 7:
            print("classyfire classes of inchikey " + full_inchikey + " at position " + str(i) + " is not complete")
            print(inchikey_results_list[i])
            inchikey_results_list[i] = [inchikey14, smiles, "", "", "", "", ""]
    #     select NPC classes
        for full_inchikey, smiles in inchikey_dict[inchikey14]:
            result = do_url_request(f"https://npclassifier.ucsd.edu/classify?smiles={smiles}")
            if result is not None:
                npc_results = get_json_npc_results(result)
                inchikey_results_list[i] += npc_results
                break
        if len(inchikey_results_list[i]) != 11:
            print("inchikey " + full_inchikey + " at position " + str(i) + " is not complete")
            print(inchikey_results_list[i])
            inchikey_results_list[i] = inchikey_results_list[i][:7] + ["", "", "", ""]
    return inchikey_results_list


def write_header(out_file: str) -> str:
    """Write classes to out_file, returns out_file with possible .txt added
    :param out_file: location of output file
    """
    if not out_file.endswith('.txt'):
        out_file += '.txt'

    header_list = [
        'inchikey', 'smiles', 'cf_kingdom',
        'cf_superclass', 'cf_class', 'cf_subclass', 'cf_direct_parent',
        'npc_class_results', 'npc_superclass_results', 'npc_pathway_results',
        'npc_isglycoside']
    with open(out_file, 'w') as outf:
        outf.write("{}\n".format('\t'.join(header_list)))
    return out_file


def write_class_info(class_info: List[List[str]],
                     out_file: str):
    """Write classes to out_file
    :param inchikey: inchikey
    :param class_info: list [smiles, cf_classes, npc_classes]
    :param out_file: location of output file
    """
    write_header(out_file)
    for row in class_info:
        with open(out_file, 'a') as outf:
            outf.write("{}\n".format('\t'.join(row)))


if __name__ == "__main__":
    from ms2query.utils import load_matchms_spectrum_objects_from_file
    # spectra = load_matchms_spectrum_objects_from_file("../../data/test_dir/test_spectra/test_spectra.mgf")
    spectra = load_matchms_spectrum_objects_from_file("../../data/Backup_MS2Query_models_paper/gnps_15_12_2021/in_between_files/ALL_GNPS_15_12_2021_negative_annotated.pickle")
    inchikey_dict = select_smiles_and_full_inchikeys(spectra)
    compound_classes = select_compound_classes(inchikey_dict)
    print(compound_classes)
    write_class_info(compound_classes, "../../data/test_dir/compound_classes.txt")

