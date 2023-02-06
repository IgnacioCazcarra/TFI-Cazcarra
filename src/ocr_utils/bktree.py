import json
import jellyfish
import pybktree


class Item():
    def __init__(self, value):
        self.value = value

        
def item_distance(x, y):
    return jellyfish.levenshtein_distance(x.value, y.value)


def english_tree():
    with open("./words_dictionary.json", "rb") as f:
        english_dict = f.read()
    english_words = json.loads(english_dict).keys()
    return pybktree.BKTree(item_distance, [Item(w) for w in english_words])


def spanish_tree():
    with open("./words_spanish.txt", "rb") as f:
        spanish_words = f.readlines()
    spanish_words = [s.decode('utf8').replace("\n", "") for s in spanish_words]
    return pybktree.BKTree(item_distance, [Item(w) for w in spanish_words])


def get_tree(mode):
    if mode == "english":
        tree = english_tree()
    else:
        tree = spanish_tree()
    return tree