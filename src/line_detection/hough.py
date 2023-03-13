import math
import random
import itertools
import cv2
import logging
import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import display
from copy import deepcopy
from ast import literal_eval

logging.basicConfig(level = logging.INFO)


def get_tablas(img_to_search, where_to_search):    
    COORDS = ['xmin','ymin','xmax','ymax']
    
    tablas = None
    cardinalidades = None
    
    for df_path in where_to_search:
        df = pd.read_csv(df_path)
        contains_img = df['image_path'].str.contains(img_to_search)
        
        if df[contains_img].shape[0] > 0:
            tablas = df[contains_img & (df['label']=="tabla")][COORDS].values
            cardinalidades = df[contains_img & (df['label']!="tabla")][COORDS].values
            return tablas, cardinalidades
    return tablas, cardinalidades


def poly_detection(img, tablas, cardinalidades):
    input_image = np.array(img)
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    #edges = cv2.Canny(input_image, 150, 200, apertureSize = 3)
    ret, thresh = cv2.threshold(input_image_gray, 190, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edges = np.zeros(input_image.shape, dtype = np.uint8) # Creamos una imagen en negro
    edges = cv2.drawContours(edges, contours, -1, (255, 255, 255), -1)
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    
    minLineLength = 5
    maxLineGap = 19
    for t in tablas:
        edges = cv2.rectangle(edges, t[:2], t[2:], color=(0,0,0), thickness=-1)
    for c in cardinalidades:
        edges = cv2.rectangle(edges, c[:2], c[2:], color=(0,0,0), thickness=-1)
#     display(Image.fromarray(edges))
    lines = cv2.HoughLinesP(edges,rho=cv2.HOUGH_PROBABILISTIC, theta=np.pi/180, threshold = 5, \
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is None or len(lines)==0:
        logging.error("No se encontraron conexiones. Salteando...")
        lines = []
    return lines, input_image


def apply_hough(img, tablas, cardinalidades):
    lines, img1 = poly_detection(img, tablas, cardinalidades)
    all_lines = {}
    for x in range(0, len(lines)):
        for i, (x1,y1,x2,y2) in enumerate(lines[x]):
            pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
            cv2.polylines(img1, [pts], True, (0,255,0))
            all_lines[str(x)+str(i)] = [[x1, y1 ], [x2 , y2]]
    return Image.fromarray(img1), all_lines


def nearest_neighbour(p, points):
    dist_dict = {k: math.dist(p, v) for k,v in points.items()}
    return min(dist_dict, key=dist_dict.get),  min(dist_dict.values())


def plot_lines(img, lines):
    img = np.array(img)
    for l in lines:
        pts = np.array(l, np.int32)
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.polylines(img, [pts], False, random_color, 2)
    return Image.fromarray(img)


def plot_points(img, points):
    img = np.array(img)
    for p in points:
        pts = np.array([p,p], np.int32)
        random_color = (255,0,0)#(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.polylines(img, [pts], True, random_color, 2)
    return Image.fromarray(img)


def lines_to_points(lines, max_dst_per_points=5):
    for k,v in lines.items():
        dst_line = math.dist(v[0],v[1])
        parts = dst_line // max_dst_per_points
        lines[k] = np.linspace(v[0], v[1], int(parts)+1).tolist()
    
    all_points = sorted([item for sublist in lines.values() for item in sublist])
    all_points = list(all_points for all_points,_ in itertools.groupby(all_points))
    all_points = {i: v for i,v in enumerate(all_points)}
    return all_points


def create_line(new_line, without_current, dst_threhold=20):
    flag = False
    to_delete = []
    while not flag:
        last_point = new_line[-1]
        nearest_point, dst = nearest_neighbour(last_point, without_current)
        
        if dst <= dst_threhold:
            new_line.append(without_current[nearest_point])
            deleted = without_current.pop(nearest_point)
            to_delete.append(nearest_point)
        else:
            new_line, to_delete2 = add_from_end(current_line=new_line, without_current=without_current,
            dst_threhold=dst_threhold)
            for deleted_key in to_delete2:
                without_current.pop(deleted_key)
            to_delete += to_delete2
            flag = True 
            
        if len(without_current) <= 1:
            flag = True
    return new_line, to_delete


def add_from_end(current_line, without_current, dst_threhold):
    ''' 
    Le agrego los posibles elementos que sean parte de la linea pero quedan colgados.
    '''
    flag = False
    to_delete = []
    without_current_2 = deepcopy(without_current)
    while not flag:
        first_point = current_line[0]
        nearest_point, dst = nearest_neighbour(first_point, without_current_2)

        if dst <= dst_threhold:
            current_line.insert(0, without_current_2[nearest_point])
            deleted = without_current_2.pop(nearest_point)
            to_delete.append(nearest_point)
        else:
            flag = True 

        if len(without_current_2) <= 1:
            flag = True
    return current_line, to_delete


def hough_detecting(all_points, **kwargs):
    if not all_points:
        return []
    
    final_lines = []
    main_flag = False
    i = 0

    while not main_flag:
        if i == 0:
            n, p = next(iter(all_points.items()))
            all_points_aux = deepcopy(all_points)
            all_points_aux.pop(n)
        else:
            n, p = next(iter(all_points_aux.items()))
            all_points_aux.pop(n)

        new_line = [p]
        without_current = deepcopy(all_points_aux)
        new_line, to_delete = create_line(new_line=new_line, without_current=without_current, **kwargs)
        
        final_lines.append(new_line)
        for deleted_key in to_delete:
            all_points_aux.pop(deleted_key)

        main_flag = len(all_points_aux.keys()) <= 1
        i += 1
    return final_lines


def point_inside(bbox, point):
    x_inside = point[0] >= bbox[0] and point[0] <= bbox[2]
    y_inside = point[1] >= bbox[1] and point[1] <= bbox[3]
    return (x_inside and y_inside)


def any_points_inside(bbox, points):
    return any(point_inside(bbox, point) for point in points)


def bbox_sum_n(bbox, n=5):
    offset = np.array([-n, -n, n, n]).reshape(1,4)
    bbox = np.sum([bbox, offset])
    return bbox.reshape(4,)


def reverse_dict(dict_cardinalidades):
    new_dict_cardinalidades = {} 
    for k,v in dict_cardinalidades.items():
        if v not in new_dict_cardinalidades.keys(): 
            new_dict_cardinalidades[v] = k
        else: 
            new_dict_cardinalidades[v] = new_dict_cardinalidades[v] + "|" + k
    return new_dict_cardinalidades


def plot_results(img, dict_cardinalidades, dict_lines):
    img = np.array(img)
    
    for k in dict_lines.keys():
        pts = np.array(dict_lines[k], np.int32)
        if k in dict_cardinalidades.keys():
            cardinalidades = dict_cardinalidades[k]
            cardinalidades = [literal_eval(card) for card in cardinalidades.split("|")]
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.polylines(img, [pts], False, random_color, 3)
            for c in cardinalidades:
                c = c[0]
                cv2.rectangle(img, c[:2], c[2:], random_color, 3)
    return Image.fromarray(img)


def unify_cardinalidades(img, lines, cardinalidades, plot=False):
    if not lines:
        return {}
    dict_cardinalidades = {}
    dict_lines = {f"line_{i}":l for i,l in enumerate(lines) if len(l)>1}
    matches = 0
    for c in cardinalidades:
        augment = 0
        offset = 1
        flag = False
        while not flag:
            c_offset = bbox_sum_n(c, offset*augment).tolist()
            for k, l in dict_lines.items():
                start = l[0]
                end = l[-1]
                if point_inside(c_offset, start) or point_inside(c_offset, end):
                    # Table + augment
                    match_key= (c.tolist(), augment)
                    dict_cardinalidades[str(match_key)] = k
                    matches +=1
                    flag = True
                    break
            if str(c.tolist()) not in dict_cardinalidades.keys():
                augment += 2
                #print(f"Increasing offset to {augment}")
    new_dict_cardinalidades = reverse_dict(dict_cardinalidades)
    if plot:
        display(plot_results(img, new_dict_cardinalidades, dict_lines))
    return new_dict_cardinalidades


def clean_cardinalidades(cardinalidades, tablas, distance_threshold=30):
    # Si tiene una tabla a menos de X puntos de distancia..
    return [c for c in cardinalidades if nearest_tabla_from_cardinalidad(c, tablas, distance_threshold=distance_threshold)]


def find_lines(tablas, cardinalidades, img, offset_tablas=5, **kwargs):
    offset = np.array([-offset_tablas, -offset_tablas, offset_tablas, offset_tablas]).reshape(1,4)
    tablas = np.sum([tablas, offset])
    img, all_lines = apply_hough(img, tablas, [])
    all_points = lines_to_points(all_lines)
    lines = hough_detecting(all_points)
    cardinalidades = clean_cardinalidades(cardinalidades, tablas)
    return unify_cardinalidades(img, lines, cardinalidades, **kwargs)


def line_to_points(x, y, max_dst_per_points=2):
    dst_line = math.dist(x,y)
    parts = dst_line // max_dst_per_points
    return np.linspace(x, y, int(parts)+1).tolist()


def dist_func(comb):
    return math.dist(comb[0], comb[1])


def get_centroid(cardinalidad):
    center_x  = cardinalidad[0] + (cardinalidad[2] - cardinalidad[0])/2
    center_y  = cardinalidad[1] + (cardinalidad[3] - cardinalidad[1])/2
    return (center_x, center_y)


def nearest_tabla_from_cardinalidad(cardinalidad, tablas, distance_threshold=999999):
    cardinalidad = literal_eval(cardinalidad) if isinstance(cardinalidad, str) else cardinalidad
    cardinalidad_centroid = get_centroid(cardinalidad)
    dict_dist_tablas = {}
    for t in tablas:
        tabla_points = []
        # De x1y1 a x2y1
        tabla_points += line_to_points((t[0], t[1]), (t[2], t[1]))
        # De x1y1 a x1y2
        tabla_points += line_to_points((t[0], t[1]), (t[0], t[3]))
        # De x1y2 a x2y2
        tabla_points += line_to_points((t[0], t[3]), (t[2], t[3]))
        # De x2y1 a x2y2
        tabla_points += line_to_points((t[2], t[1]), (t[2], t[3]))
        dist_cardinalidad = [(cardinalidad_centroid, point) for point in tabla_points]
        min_combination = min(dist_cardinalidad, key=dist_func)
        dict_dist_tablas[",".join([str(c) for c in t])] = dist_func(min_combination)
    nearest_tabla = min(dict_dist_tablas, key=dict_dist_tablas.get)
    
    if min(dict_dist_tablas.values()) >= distance_threshold:
        return [] 
    return [int(c) for c in nearest_tabla.split(",")]


def sep_line(line, tablas):
    tabla_a = None
    tabla_b = None
    try:
        cardinalidades = line.split("|")
        cardinalidades = [literal_eval(c) for c in cardinalidades]
        cardinalidades_dist = {str(c[0]): c[1] for c in cardinalidades}
        # TOP 2 con menos augment
        cardinalidades = sorted(cardinalidades_dist, key=cardinalidades_dist.get)[:2]
        tabla_a = nearest_tabla_from_cardinalidad(cardinalidades[0], tablas)
        tabla_b = nearest_tabla_from_cardinalidad(cardinalidades[1], tablas)
    except Exception as e:
        logging.warning(f"Error al separar tablas! {e}. Chequear las bounding boxes pasadas. Salteando..")
    finally:
        return (tabla_a, tabla_b)

    
def get_pairs(boxes_tablas, boxes_cardinalidades, img, **kwargs):
    pairs = []
    tablas = boxes_tablas.detach().numpy().astype(int)
    cardinalidades = boxes_cardinalidades.detach().numpy().astype(int)
        
    for line_name, line in find_lines(img=img, tablas=tablas, cardinalidades=cardinalidades, **kwargs).items(): 
        tabla_a, tabla_b = sep_line(line, tablas)
        if tabla_a and tabla_b:
            pairs.append((tabla_a, tabla_b))
    return pairs