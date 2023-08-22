from .bktree import get_tree, Item
from .. import constants
import re
import cv2
import spacy
import logging
import jellyfish
import lemminflect
import numpy as np
import itertools
from paddleocr import PaddleOCR
from more_itertools import subslices
from unidecode import unidecode

logging.basicConfig(level = logging.INFO)


def get_ocr_model():
    ocr = PaddleOCR(ocr_version='PP-OCRv3', use_angle_cls=False, show_log=False, det_db_score_mode="slow", lang="en")
    return ocr


def get_lemmatizer(lang="en"):
    return spacy.load("es_core_news_sm") if lang == "es" else spacy.load("en_core_web_sm")


def get_plural(word, lemmatizer):
    return lemmatizer(word)[0]._.inflect('NNS')


def extract_from_ocr(coords, results, **kwargs):
    all_tables = {}
    tables_names = {}
    for c, r in zip(coords, results):
        boxes = [line[0] for line in r]
        txts = [line[1][0].strip() for line in r]
        scores = [line[1][1] for line in r]
        table, dict_attributes = clean_texts(txts, **kwargs)
        if not table and not dict_attributes:
            logging.info(f"Removing table in coordinates {c}")
            # In that case the table was as FP. Need to skip it.
            continue
        all_tables[table] = dict_attributes
        tables_names[table] = c
    return all_tables, tables_names


def predict_ocr(img, tablas, ocr_model, scale_percent=100, lang="en"):
    coords = []
    results = []
    img_arr = img if isinstance(img, np.ndarray) else np.array(img)
    for t in tablas:
        tabla_cropped = img_arr[t[1]:t[3], t[0]:t[2]]
        tabla_cropped = reescale(tabla_cropped, scale_percent)
        if len(tabla_cropped.shape) == 3 and tabla_cropped.shape[-1] > 3:
            tabla_cropped = tabla_cropped[:,:,:3]
        result = ocr_model.ocr(tabla_cropped, cls=False)

        coords.append(t.tolist())
        results.append(result[0])
    return extract_from_ocr(coords=coords, results=results, mode=lang)


def pairs_to_names(pairs, tables_names):
    new_pairs = []
    tables_names = {str(v):k for k,v in tables_names.items()}
    
    for pair in pairs:
        try:
            tabla_a, tabla_b = pair
            tabla_a, tabla_b = tables_names[str(tabla_a)], tables_names[str(tabla_b)]
            new_pairs.append((tabla_a, tabla_b))
        except KeyError as e:
            logging.warn(f"Skipping pair of tables {str(tabla_a)} {str(tabla_b)} because one is a false positive.")
    return new_pairs


def _is_potential_key(table, attribute, other_tables, lemmatizer):
    if "_id" in attribute or attribute == "id" or attribute in generate_valid_combs_fk(table, lemmatizer):
        return True 
    for t in other_tables:
        other_tables_valid_combs = generate_valid_combs_fk(t, lemmatizer)
        if attribute in other_tables_valid_combs:
            return True
    return False


def extract_candidate_keys(table, table_attributes, other_tables, lemmatizer):
    return [attr for attr in table_attributes if _is_potential_key(table, attr, other_tables, lemmatizer)]


def initial_guess_primary_keys(table, candidates, lemmatizer):
    table_lemmatized = lemmatizer(table)[0].lemma_
    table_unlemmatized = get_plural(table, lemmatizer)
    possibilities = ["id", table+"_id", table_lemmatized+"_id", table+"id", table_lemmatized+"id", table_unlemmatized+"id", table_unlemmatized+"_id"]
    return list(set([p for p in possibilities if p in candidates]))


def is_many_to_many(table, table_candidates, tables_names, pairs, lemmatizer):
    # Con esto deberian quedar solo dos pero hay que preever que pasaría si hay más.
    matches = [t for t in tables_names if ((t in table) or (lemmatizer(t)[0].lemma_ in table) or (get_plural(t, lemmatizer) in table)) and (t!=table)]
    i = 0
    confirmed_matches = []
    flag = False
    while not flag and i<len(pairs):
        pair = pairs[i]
        if table not in pair:
            i+=1
            continue

        if pair[0] in matches:
            # Confirmamos que hay una tabla que aparece en el nombre de la m2m y
            # tiene relación con ella.
            confirmed_matches.append(pair[0]) 
        elif pair[1] in matches:
            confirmed_matches.append(pair[1])
            
        if len(confirmed_matches) == 2:
            # Si hay dos tablas que aparecen en el nombre de la m2m y tienen conexión con ella, es confirmado
            flag = True
        i+=1
    return flag


def get_unchosen(candidates, already_chosen):
    '''
    Remueve los candidatos ya elegidos
    '''
    return list(set(candidates) - set(already_chosen))


def generate_valid_combs_fk(table, lemmatizer):
    '''
    Dado el nombre de una tabla, genera las combinaciones válidas. Osea, tabla+id, tabla+_id, 
    tabla_lematizada+id y tabla_lematizada+_id. También aplica a casos donde la tabla está en singular y 
    la PK en plural.
    '''
    table_lemmatized = lemmatizer(table)[0].lemma_
    table_unlemmatized = get_plural(table, lemmatizer)
    valid_combs = [table+"id", table+"_id", 
                   table_lemmatized+"id", table_lemmatized+"_id",
                   table_unlemmatized+"id", table_unlemmatized+"_id"]
    return list(set(valid_combs))


def match_fk(table_pair, table_candidates, table_pair_candidates, lemmatizer):
    '''
    Match normal entre dos variantes con el nombre de la tabla.
    '''
    valid_combs = generate_valid_combs_fk(table_pair, lemmatizer)
    possibilities = valid_combs
    pair_possibilities = valid_combs + ["id"]
    
    for possibility in possibilities:
        for pair_possibility in pair_possibilities:
            if possibility in table_candidates and pair_possibility in table_pair_candidates:
                table_candidates.remove(possibility) # Remuevo la fk de la lista de candidatos.
                return (True, possibility, pair_possibility)
    return (False, "", "")


def match_autofk(table, table_candidates, lemmatizer):
    valid_combs = generate_valid_combs_fk(table, lemmatizer) + ["id"]
    table_lemmatized = lemmatizer(table)[0].lemma_
    table_unlemmatized = get_plural(table, lemmatizer)
    fk = None
    pk = None
    
    i = 0
    while not fk and i<len(table_candidates):
        t = table_candidates[i]
        if ((table in t) or (table_lemmatized in t) or (table_unlemmatized in t)) and (t not in valid_combs):
            fk = t
        i+=1
    j = 0
    while not pk and j<len(valid_combs):
        v = valid_combs[j]
        if v in table_candidates:
            pk = v
        j+=1
    
    if pk:
        if fk:
            return (True, fk, pk)
        else:
            left_candidates = get_unchosen(table_candidates, valid_combs)
            if len(left_candidates) == 1:
                return (True, left_candidates[0], pk)
    return (False, "", "")


def match_m2m(table, table_pair, table_candidates, table_pair_candidates, lemmatizer):
    '''
    Chequea si hay un atributo en común entre la tabla normal y la m2m. Si hay uno solo, se devuelve ese.
    Si no, se sigue con la opción "normal" entre dos tablas convencionales (método 'match_fk').
    '''
    matches = [table_candidate for table_candidate in table_candidates if table_candidate in table_pair_candidates]
    if len(matches) == 1:
        # Si hubo match con solo un atributo.
        # A las m2m no se les remueve la FK porque tambien es PK. 
        return (True, matches[0], matches[0])
    else:
        # Si no hubo un solo match, hacemos el chequeo "normal" con las combinaciones válidas de la tabla.
        return match_fk(table_pair, table_candidates, table_pair_candidates, lemmatizer)


def is_foreign_key(table, table_pair, table_candidates, table_pair_candidates, lemmatizer, m2m_tables,
                   is_auto_fk=False):
    '''
    Se fija si hay un match entre un atributo con _id en su versión original y lematizada.
    Soporta relaciones convencionales, relaciones many to many y auto foreign keys.
    '''
    if table_pair == table and not is_auto_fk:
        return (False, "", "")

    if is_auto_fk:
        return match_autofk(table, table_candidates, lemmatizer)
    elif table in m2m_tables:
        # Si es una relación y la tabla es una "many to many"
        return match_m2m(table, table_pair, table_candidates, table_pair_candidates, lemmatizer)
    else:
        # Si es una relación entre dos tablas "estándar".
        return match_fk(table_pair, table_candidates, table_pair_candidates, lemmatizer)



def get_foreign_keys(table, all_candidates, pairs, pairs_labels, m2m_tables, lemmatizer, check_auto_fks=False):
    """
    Ejemplo:
    table -> poems
    candidates -> ['poems_id', 'users_id', 'categories_id']
    pairs -> [('tokens', 'users'), ('poems', 'users'), ('poems', 'categories')]
    """
    fks = {}
    completed_pairs = []
    le_dict_cardinalidades = {v:k for k,v in constants.le_dict_cardinalidades.items()} # reverse it

    table_candidates = all_candidates[table]
    for pair, pair_label in zip(pairs, pairs_labels):
        if table not in pair:
            continue
        is_auto_fk = False
        if pair[0] == pair[1] and check_auto_fks:
            is_auto_fk = True
        is_fk_pair0, table_att0, pair_att0 = is_foreign_key(table=table, table_pair=pair[0], 
                                                            table_candidates=table_candidates,
                                                            table_pair_candidates=all_candidates[pair[0]], 
                                                            lemmatizer=lemmatizer, is_auto_fk=is_auto_fk,
                                                            m2m_tables=m2m_tables)
        is_fk_pair1, table_att1, pair_att1 = is_foreign_key(table=table, table_pair=pair[1], 
                                                            table_candidates=table_candidates,
                                                            table_pair_candidates=all_candidates[pair[1]], 
                                                            lemmatizer=lemmatizer, is_auto_fk=is_auto_fk,
                                                            m2m_tables=m2m_tables)
        
        if is_fk_pair0:
            fks[(table_att0, pair_att0)] = (pair[0], le_dict_cardinalidades[pair_label[0]+1])
            completed_pairs.append(pair)
        elif is_fk_pair1:
            fks[(table_att1, pair_att1)] = (pair[1], le_dict_cardinalidades[pair_label[1]+1])
            completed_pairs.append(pair)
    return fks, completed_pairs


def generate_pks_code(pks):
    keys = pks.keys()
    keys = ", ".join(['`'+ k +'`' for k in keys])
    if not keys:
        return ""
    return f"PRIMARY KEY ({keys})"


def generate_fks_code(table, fks):
    code = ""
    for fk, table_reference in fks.items():
        code += f"ALTER TABLE `{table}` ADD FOREIGN KEY (`{fk[0]}`) REFERENCES `{table_reference[0]}` (`{fk[1]}`); \n"
    return code


def get_fks_att_type(fks):
    attributes = {}
    for fk, table_reference in fks.items():
        if "opcional" in table_reference[1]:
            attributes[fk[0]] = " " + "NULL"
        else:
            attributes[fk[0]] = " " + "NOT NULL"
    return attributes


def create_code(table, dict_attributes, primary_keys, foreign_keys):
    '''
    Crea una tabla de MySQL
    '''
    attributes_code = "  "
    fks_type = get_fks_att_type(fks=foreign_keys)
    i = 0
    for k, v in dict_attributes.items():
        attributes_code += "`" +k+ "`" + " " + v
        if k in primary_keys.keys():
            attributes_code += " " + "NOT NULL"
        else:
            attributes_code += fks_type.get(k, '')
        attributes_code += ",\n   "
        i += 1
    pks_code = generate_pks_code(primary_keys)
    fks_code = generate_fks_code(table, foreign_keys)
    if pks_code:
        attributes_code += pks_code
    else:
        # Remove ",\n   "
        attributes_code = attributes_code[:-5]
    code = f" CREATE TABLE `{table}` ( \n {attributes_code} \n ); \n"
    return code, fks_code


def print_remaining_pairs(pairs):
    for p in pairs:
        logging.warning(f"No se pudo establecer la relación entre {p}. Por favor, chequear que los atributos estén en el formato correcto.\n") 


def generate_db(pairs, pairs_labels, all_tables, tables_names, lang):
    pairs = pairs_to_names(pairs, tables_names)
    lemmatizer = get_lemmatizer(lang=lang)
    all_candidates = {}
    m2m_tables = []
    # Primera pasada: Extraemos los candidatos y vemos qué tabla es m2m.
    for k, dict_attributes in all_tables.items():
        candidates = extract_candidate_keys(table=k, table_attributes=dict_attributes.keys(), 
                                            other_tables=[t for t in tables_names if t != k], lemmatizer=lemmatizer)
        all_candidates[k] = candidates
        if is_many_to_many(k, candidates, tables_names, pairs, lemmatizer=lemmatizer):
            m2m_tables.append(k)
    all_tables_pks = {}
    all_tables_fks = {}
    # Segunda pasada: Se resuelven todas las relaciones menos la de auto fks.
    for k in all_tables.keys():
        pks = {pk: k for pk in initial_guess_primary_keys(k, all_candidates[k], lemmatizer)}
        fks, completed_pairs = get_foreign_keys(table=k, all_candidates=all_candidates, pairs=pairs, pairs_labels=pairs_labels,\
                                                m2m_tables=m2m_tables, check_auto_fks=False, lemmatizer=lemmatizer)
        pairs = get_unchosen(pairs, completed_pairs)
        pks = {**pks, **{pk: k for pk in get_unchosen(all_candidates[k], fks.keys())}}
        all_tables_pks[k] = pks
        all_tables_fks[k] = fks
    all_code = ""
    all_fks_code = ""
    # Tercera pasada: Se completan los auto-fks y se genera el código.
    for k, dict_attributes in all_tables.items():
        fks, completed_pairs = get_foreign_keys(table=k, all_candidates=all_candidates, pairs=pairs, pairs_labels=pairs_labels,\
                                                m2m_tables=m2m_tables, check_auto_fks=True, lemmatizer=lemmatizer)
        pairs = get_unchosen(pairs, completed_pairs)
        if fks:
            all_tables_fks[k] = {**all_tables_fks[k], **fks}
            
        code, fk_code = create_code(k, dict_attributes, \
                                    primary_keys=all_tables_pks[k], \
                                    foreign_keys=all_tables_fks[k])
        all_code += code
        all_fks_code += fk_code
    print_remaining_pairs(pairs)
    return all_code + "\n" + all_fks_code


def reescale(img, scale_percent):
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
        logging.error(f"'{db_name}' not supported yet!")
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
            max_key_score = c
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
        
        
def sanitize_words(splitted_attribute, mode="en", **kwargs):
    '''
    Sanitizes every word of the attribute.
    '''
    tree = get_tree(mode)
    
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
                # Me quedo con la secuencia en partes cuya suma de scores sea la mayor.
                fixed_word = get_best_performing_sequence(word, slices_dict)
            sanitized.append(fixed_word)
        else:
            sanitized.append(word) # Dejar así nomás; no suele haber typos y la podes cagar facilmente.
    return "_".join(sanitized)


def get_clean_attribute(attribute, **kwargs):
    # Correct common error for the attributes. Doesn't affect results.
    attribute = attribute.strip().replace("I","l")
    if " " in attribute:
        splitted_attribute = attribute.split("_")
        attribute = sanitize_words(splitted_attribute, **kwargs)
    return unidecode(attribute) 


def get_valid_table_att(text_list):
    flag = False
    valid = ""
    i = -1
    while not flag and i < len(text_list)-1:
        i += 1
        if text_list[i] == "|":
            # Correct common mistake.
            valid += "l"
        elif not text_list[i].isalpha() and not valid:
            continue
        elif text_list[i].isupper() and not valid:
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


def separate(text, db_name="mysql", **kwargs):
    text_list = sep_text(text)
    text_list = " ".join(text_list)
    attribute, dtype = delim_attribute(text_list)
    attribute = get_clean_attribute(attribute, **kwargs)
    dtype_number = get_dtype_number(dtype)
    dtype = dtype.replace("(", "").replace(")","")
    dtype, _ = get_dtype(dtype, dtypes=get_allowed_dtypes(db_name))
    if dtype_number:
        dtype += f"({dtype_number[0]})"
    return attribute, dtype


def rename_duplicated_attribute(attributes, attribute):
    suffix = 1
    new_key = attribute
    while new_key in attributes:
        new_key = f"{attribute}_{suffix}"
        suffix += 1
    return new_key


def clean_texts(texts, **kwargs):
    if not texts or len(texts) <= 1:
        logging.error("No se encontró texto para una de las tablas. Salteando...")
        return "", {}
    
    if "Indexes" in texts:
        # Todo lo que venga después de Indexes está mal o pertenece a otra cosa.
        indexes_idx = texts.index("Indexes")
        texts = texts[:indexes_idx]
    table_name, _ = get_valid_table_att(texts[0].lower()) # Para limpiarlo por si tiene espacios o simbolos.
    table_name = get_clean_attribute(table_name, **kwargs) # Para insertarle "_" en caso de que haya falta.
    attributes = {} # K=name, V=type
    for t in texts[1:]:
        if len(t.strip()) == 1:
            continue
        attribute, dtype = separate(t, db_name="mysql", **kwargs)
        if attribute.strip() not in attributes.keys():
            attributes[attribute.strip()] = dtype
        else:
            new_key = rename_duplicated_attribute(attributes, attribute.strip())
            logging.warning(f" {attribute.strip()} ({dtype}) is already in the table. Renaming it to {new_key}..")
            attributes[new_key] = dtype
    return table_name, attributes
