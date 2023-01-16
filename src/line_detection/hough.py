import os
import math
import random
import itertools
from PIL import Image
from copy import deepcopy
from ast import literal_eval

import cv2
import numpy as np
import pandas as pd


def get_tablas(img_to_search):
    train = pd.read_csv("../data/csv/train_diagramas.csv")
    val = pd.read_csv("../data/csv/val_diagramas.csv")
    test = pd.read_csv("../data/csv/test_diagramas.csv")
    
    COORDS = ['xmin','ymin','xmax','ymax']
    
    tablas = None
    cardinalidades = None
    train_contains_img = train['image_path'].str.contains(img_to_search)
    test_contains_img = test['image_path'].str.contains(img_to_search)
    val_contains_img = val['image_path'].str.contains(img_to_search)
    
    if train[train_contains_img].shape[0] > 0:
        tablas = train[train_contains_img & (train['label']=="tabla")][COORDS].values
        cardinalidades = train[train_contains_img & (train['label']!="tabla")][COORDS].values
    elif val[val_contains_img].shape[0] > 0:
        tablas = val[val_contains_img & (val['label']=="tabla")][COORDS].values
        cardinalidades = val[val_contains_img & (val['label']!="tabla")][COORDS].values
    else:
        tablas = test[test_contains_img & (test['label']=="tabla")][COORDS].values
        cardinalidades = test[test_contains_img & (test['label']!="tabla")][COORDS].values

    return tablas, cardinalidades


def poly_detection(img_path, tablas, cardinalidades):
    inputImage = cv2.imread(img_path)
    inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    
    #edges = cv2.Canny(inputImageGray, 150, 200, apertureSize = 3)
    ret, thresh = cv2.threshold(inputImageGray, 190, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    edges = np.zeros(inputImage.shape, dtype = np.uint8) # Creamos una imagen en negro
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
    
    return lines, inputImage


def apply_hough(img_path, tablas, cardinalidades):
    lines, img1 = poly_detection(img_path, tablas, cardinalidades)
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
    for p in points.values():
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
            cv2.polylines(img, [pts], False, random_color, 2)
            for c in cardinalidades:
                cv2.rectangle(img, c[:2], c[2:], random_color, 2)
    return Image.fromarray(img)


def unify_cardinalidades(img, lines, cardinalidades):
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
                    dict_cardinalidades[str(c.tolist())] = k
                    matches  +=1
                    flag = True
                    break
            if str(c.tolist()) not in dict_cardinalidades.keys():
                augment += 2
                print(f"Increasing offset to {augment}")

    new_dict_cardinalidades = reverse_dict(dict_cardinalidades)
    return plot_results(img, new_dict_cardinalidades, dict_lines)