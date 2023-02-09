from .bktree import get_tree

import re
import os
import cv2
import json
import jellyfish
import pybktree
import numpy as np
import itertools
from more_itertools import subslices

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from IPython.display import display


def reescale(img, scale_percent=150):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    return img


def get_allowed_dtypes(db_name):
    if db_name.lower() == "mysql":
        STRING_TYPES = ["CHAR", "VARCHAR", "BINARY", "VARBINARY", "TINYBLOB", "TINYTEXT", "TEXT", "BLOB", 
                        "MEDIUMTEXT", "MEDIUMBLOB", "LONGTEXT", "LONGBLOB", "ENUM", "SET"]
        NUMERIC_TYPES = ["BIT", "TINYINT", "BOOLEAN", "SMALLINT", "MEDIUMINT", "INT", "INTEGER", "BIGINT", 
                         "FLOAT", "DOUBLE", "DOUBLE PRECISION", "DECIMAL", "DEC"]
        DATETIME_TYPES = ["DATE", "DATETIME", "TIMESTAMP", "TIME", "YEAR"]
        SPATIAL_TYPES = ["GEOMETRY", "POINT", "LINESTRING", "POLYGON"]
        return (STRING_TYPES + NUMERIC_TYPES + DATETIME_TYPES + SPATIAL_TYPES)
    else:
        print(f"'{db_name}' not supported yet!")
        return []
    
    
def get_dtype_number(text):
    if "," not in text:
        return re.findall("\d+", text) # Devuelve solo numeros juntos.
    else:
        return re.findall("\d+\,\d+", text) # Devuelve numeros con coma para dtypes como DECIMAL(5,2)
    
    
def sep_text(text):
    '''
    Separa las palabras pegadas a una mayúscula como city_idVARCHAR(45). O sea, cuando no se detecta el espacio.
    '''
    return re.findall("[A-Z]+[^A-Z]*|[^A-Z]+",text)


def get_dtype(possible_dtype, dtypes):
    '''
    Devuelve el dtype que más se le parece al string. Para hacerlo más rapido podría usar un bktree.
    '''
    dict_dtypes = {k: jellyfish.jaro_distance(possible_dtype, k) for k in dtypes}
    return (max(dict_dtypes, key=dict_dtypes.get), max(dict_dtypes.values()))


def chunks(items, cutpoints):
    return [items[i:j] for i,j in zip([0] + cutpoints, cutpoints + [len(items)])]


def generate_chunks(items, n):
    indices = range(1,len(items))
    return [chunks(items,list(cutpoints)) for cutpoints in itertools.combinations(indices,n-1)]


def get_average_score(comb, slices_dict):
    scores = []
    word = ""
    for c in comb:
        dict_c = slices_dict[c]
        if dict_c:
            max_key_score = max(dict_c, key=dict_c.get)
            max_score = dict_c[max_key_score]
        else:
            max_key_score = ""
            max_score = 0
        scores.append(max_score)
        word += max_key_score + "_"
    return word[:-1], sum(scores)/len(scores), len(comb)


def get_best_performing_sequence(word, slices_dict):
    all_combs = []
    for i in range(len(word)):
        all_combs += generate_chunks(word, i+1)
    all_combs = [[el.replace(" ", "") for el in l if not el.isspace()] for l in all_combs]
    all_combs.sort()
    all_combs = [l for l in list(k for k,_ in itertools.groupby(all_combs)) if all(el in slices_dict.keys() for el in l)]
    
    combs_dict = {}
    for comb in all_combs:
        word, avg_score, comb_len = get_average_score(comb, slices_dict)
        combs_dict[word] = (avg_score, comb_len)
        
    max_key = -1
    min_length = 99999
    max_score = 0

    for k,v in combs_dict.items():
        if v[0] >= max_score and v[1] < min_length:
            max_key = k
            max_score = v[0]
            min_length = v[1]
        
    return max_key


def get_dist_map(key, neighbors, top_n=5):
    dist_map = {n[1]: round(jellyfish.jaro_distance(key, n[1]), 2) for n in neighbors}
    sorted_topn = sorted(dist_map, key=dist_map.get, reverse=True)[:top_n]
    return {k:v for k,v in dist_map.items() if k in sorted_topn}


def top_n_dist_map(tree, slice_key, tolerance, top_n):
    slice_value = tree.find(Item(slice_key), tolerance)
    if slice_value:
        slice_value = [(res[0], res[1].value) for res in slice_value]
    return get_dist_map(key=slice_key, neighbors=slice_value, top_n=top_n)


def get_slices_dict(tree, slices, tolerance=1, top_n=5):
    slices_dict = {}
    for slice_ in slices:
        slice_key = "".join(slice_)
        slice_value = top_n_dist_map(tree, slice_key, tolerance, top_n)
        slices_dict[slice_key] = slice_value
    return slices_dict
        
        
def sanitize_words(splitted_attribute, mode="english"):
    '''
    Sanitizes every word of the attribute.
    '''
    tree = get_tree(mode)
    TOLERANCE = 1
    
    sanitized = []
    for word in splitted_attribute:
        word = word.strip()
        if " " in word:
            word_to_fix = word.split(" ")
            slices = list(subslices(word_to_fix))
            slices = ["".join(slice_).replace(" ", "") for slice_ in slices if slice_]
            slices = list(set(slices))
            slices_dict = get_slices_dict(tree, slices)
            longest_key = max(slices_dict.keys(), key=len) # Agarramos la secuencia más larga.
            longest_key_dict = slices_dict[longest_key] # Top N para esa secuencia.
            # Hay que dar por hecho que no hay typos, solo problemas con los espacios.
            if longest_key_dict and max(longest_key_dict.values()) == 1:
                longest_key_max_score = max(longest_key_dict, key=longest_key_dict.get)
                fixed_word = longest_key_max_score
            else:
                # Me tengo la secuencia en partes cuya suma de scores sea la mayor.
                fixed_word = get_best_performing_sequence(word, slices_dict)
            sanitized.append(fixed_word)
        else:
            sanitized.append(word) # Dejar así nomás; no suele haber typos y la podes cagar facilmente.
    return "_".join(sanitized)


# El problema es si se confunde una L por una | o cosas así. Status: deprecated.
def clean_attribute(attribute):
    '''
    Removes from str everything that's not a digit, underscore, dollar signs & characters (upper or lower).
    '''
    return re.sub("[^0-9a-zA-Z$_]+", "", attribute)


def get_clean_attribute(attribute):
    '''
    Debería ser algo así:
    - Separa por underscore porque espacios no deberia haber.
    - Lo que haya quedado junto y tenga espacios en blanco, lo probamos con y sin y nos quedamos con la palabra de
    ambas divisiones que tenga menor tolerancia.
    - Si hay que juntar juntamos, y si no separamos.
    - Reemplazamos los espacios por underscore y devolvemos.
    '''
    attribute = attribute.strip()
    if " " in attribute:
        splitted_attribute = attribute.split("_")
        attribute = sanitize_words(splitted_attribute)
    return attribute.replace("|","l").replace("I","l")


def get_valid_table_att(text_list):
    flag = False
    valid = ""
    i = -1
    while not flag and i < len(text_list)-1:
        i += 1
        if not text_list[i].isalpha() and not valid:
            continue
        if text_list[i].isupper() and not valid:
            valid += text_list[i]
        elif text_list[i].islower() or text_list[i].isspace() or text_list[i].isdigit() or text_list[i] in ["_","$"]:
            valid += text_list[i]
        else:
            flag = True
    return valid, i


def delim_attribute(text_list): 
    attribute, i = get_valid_table_att(text_list)
    dtype = "".join(text_list[i:])
    return attribute, dtype


def separate(text, db_name="mysql"):
    text_list = sep_text(text)
    text_list = " ".join(text_list)
    attribute, dtype = delim_attribute(text_list)
    attribute = get_clean_attribute(attribute)
    dtype_number = get_dtype_number(dtype)
    dtype = dtype.replace("(", "").replace(")","")
    dtype, _ = get_dtype(dtype, dtypes=get_allowed_dtypes(db_name))
    if dtype_number:
        dtype += f"({dtype_number[0]})"
    return attribute, dtype


def clean_texts(texts):
    if "Indexes" in texts:
        # Todo lo que venga después de Indexes está mal o pertenece a otra cosa.
        indexes_idx = texts.index("Indexes")
        texts = texts[:indexes_idx]
    table_name, _ = get_valid_table_att(texts[0]) # Para limpiarlo por si tiene espacios o simbolos.
    attributes = {} # K=name, V=type
    for t in texts[1:]:
        attribute, dtype = separate(t, db_name="mysql")
        attributes[attribute.strip()] = dtype
    return table_name, attributes