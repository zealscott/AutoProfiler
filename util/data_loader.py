import json
import os


def load_synthpai(useranme):
    dict_path = f"./dataset/synthpai/{useranme}"
    ret = []
    files = os.listdir(dict_path)
    # sort the files by
    files.sort(key=lambda x: int(x.strip(".txt").split("_")[-1]))
    for file in files:
        with open(dict_path + "/" + file, "r") as f:
            ret.append(f.read())
    return ret


def check_valid(username):
    """
    in SynthPAI, not all synthetic users have ground truth
    check whether the target user has ground truth
    """
    attributes = []
    ground_truth_path = f"./dataset/ground_truth.json"
    ground_truth = json.load(open(ground_truth_path, "r"))
    if username not in ground_truth:
        print(f"no ground truth for {username}")
        return attributes

    for attr_dict in ground_truth[username]:
        attributes.append(list(attr_dict.keys())[0])
    print(f"Target attributes for {username}: {attributes}")
    return attributes


class SafeDict(dict):
    """
    this is a dictionary that returns the key if the key is missing in the dictionary
    """

    def __missing__(self, key):
        return "{" + key + "}"
