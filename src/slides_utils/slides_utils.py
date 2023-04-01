# Adaptation of the code available at https://github.com/acl21/image_bbox_slicer

from ..constants import *
from ..detection.prediction_utils import filter_predictions

import os
import math
import glob
import torch
import logging
import numpy as np
import image_bbox_slicer as ibs
from PIL import Image
from ast import literal_eval
from image_bbox_slicer.helpers import * 
from image_bbox_slicer.slicer import Points

logging.basicConfig(level = logging.INFO)


def default_ibs(img_src, ann_src, img_dst, ann_dst):
    slicer = ibs.Slicer()
    slicer.config_dirs(img_src=img_src, ann_src=ann_src, 
                       img_dst=img_dst, ann_dst=ann_dst)
    slicer.keep_partial_labels = True
    slicer.save_before_after_map = True
    return slicer


def __slice_images(slicer, tile_size, tile_overlap, number_tiles):
    mapper = {}
    img_no = 1

    for file in sorted(glob.glob(slicer.im_src + os.sep + "*")):
        file_name = file.split(os.sep)[-1].split('.')[0]
        file_type = file.split(os.sep)[-1].split('.')[-1].lower()

        im = Image.open(file)

        if number_tiles > 0:
            n_cols, n_rows = calc_columns_rows(number_tiles)
            tile_w, tile_h = int(
                math.floor(im.size[0] / n_cols)), int(math.floor(im.size[1] / n_rows))
            tile_size = (tile_w, tile_h)
#             tile_overlap = 0.0

        tiles = __get_tiles(im.size, tile_size, tile_overlap)
        new_ids = []
        for tile in tiles:
            new_im = im.crop(tile)
            img_id_str = str('{:06d}'.format(img_no))
            if len(slicer._ignore_tiles) != 0:
                if img_id_str in slicer._ignore_tiles:
                    slicer._ignore_tiles.remove(img_id_str)
                    continue
            new_im.save(
                '{}{}{}.{}'.format(slicer.im_dst, os.sep, img_id_str, file_type))
            new_ids.append(img_id_str)
            img_no += 1
        mapper[file_name] = new_ids
    return mapper


def slice_img_from_prediction(img, tile_size, tile_overlap, number_tiles):
    if number_tiles > 0:
        n_cols, n_rows = calc_columns_rows(number_tiles)
        tile_w, tile_h = int(math.floor(img.size[0] / n_cols)), int(math.floor(img.size[1] / n_rows))
        tile_size = (tile_w, tile_h)
    tiles = __get_tiles(img.size, tile_size, tile_overlap)   
    return tiles


def __get_tiles(img_size, tile_size, tile_overlap):
    """Generates a list coordinates of all the tiles after validating the values.
    Private Method.
    Parameters
    ----------
    img_size : tuple
        Size of the original image in pixels, as a 2-tuple: (width, height).
    tile_size : tuple
        Size of each tile in pixels, as a 2-tuple: (width, height).
    tile_overlap: float
        Percentage of tile overlap between two consecutive strides.
    Returns
    ----------
    list
        A list of tuples.
        Each holding coordinates of possible tiles
        in the format - `(xmin, ymin, xmax, ymax)`
    """
    validate_tile_size(tile_size, img_size)
    tiles = []
    img_w, img_h = img_size
    tile_w, tile_h = tile_size
    stride_w = int((1 - tile_overlap) * tile_w)
    stride_h = int((1 - tile_overlap) * tile_h)
    
    tile_overlap = 0.6
    tmp_stride_w = int((1 - tile_overlap) * tile_w)
    tmp_stride_h = int((1 - tile_overlap) * tile_h)
        
    for y in range(0, img_h-tile_h+1, stride_h):
        
        reached_max_h = (y + tile_h + stride_h >= img_h-tile_h)
        just_started_h = (y == 0)

        if(just_started_h):
            y = 0
            y2 = y + tile_h + tmp_stride_h
        elif(reached_max_h):
            y -= tmp_stride_h
            y2 = y + tile_h + tmp_stride_h
        else:
            y += tmp_stride_h//2
            y2 = y + tile_h + tmp_stride_h//2

        for x in range(0, img_w-tile_w+1, stride_w):            
#             x2 = x + tile_w
#             y2 = y + tile_h
#             tiles.append((x, y, x2, y2))

            reached_max_w = (x + tile_w + stride_w >= img_w-tile_w)

            just_started_w = (x == 0)

            if(just_started_w):
                x = 0
                x2 = x + tile_w + tmp_stride_w
            elif(reached_max_w):
                x -= tmp_stride_w
                x2 = x + tile_w + tmp_stride_w
            else:
                x += tmp_stride_w//2
                x2 = x + tile_w + tmp_stride_w//2
            
            tiles.append((x, y, x2, y2))                
    return tiles


def validate_tile_size(tile_size, img_size=None):
    """Validates tile size argument provided for slicing.
    Parameters
    ----------
    tile_size : tuple
        Size of each tile in pixels, as a 2-tuple: (width, height).
    img_size : tuple, optional
        Size of original image in pixels, as a 2-tuple: (width, height).
    Returns
    ----------
    None
    Raises
    ----------
    ValueError
        If `tile_size` does not hold exactly `2` values
        If `tile_size` does not comply with `img_size`
    TypeError
        If `tile_size` or `img_size` are not of type tuple.
    """
    if img_size is None:
        if isinstance(tile_size, tuple):
            if len(tile_size) != 2:
                raise ValueError(
                    'Tile size must be a tuple of size 2 i.e., (w, h). The tuple provided was {}'.format(tile_size))
        else:
            raise TypeError(
                'Tile size must be a tuple. The argument was of type {}'.format(type(tile_size)))
    else:
        if isinstance(img_size, tuple):
            if (sum(tile_size) >= sum(img_size)) or (tile_size[0] > img_size[0]) or (tile_size[1] > img_size[1]):
                raise ValueError('Tile size cannot exceed image size. Tile size was {} while image size was {}'.format(
                    tile_size, img_size))
        else:
            raise TypeError(
                'Image size must be a tuple. The argument was of type {}'.format(type(img_size)))
            
            
def __slice_bboxes(slicer, tile_size, tile_overlap, number_tiles):
    img_no = 1
    mapper = {}
    empty_count = 0

    for xml_file in sorted(glob.glob(slicer.ann_src + os.sep + '*.xml')):
        root, objects = extract_from_xml(xml_file)
        im_w, im_h = int(root.find('size')[0].text), int(
            root.find('size')[1].text)
        im_filename = os.path.splitext(root.find('filename').text)[0]
        extn = os.path.splitext(root.find('filename').text)[1]
        if number_tiles > 0:
            n_cols, n_rows = calc_columns_rows(number_tiles)
            tile_w = int(math.floor(im_w / n_cols))
            tile_h = int(math.floor(im_h / n_rows))
            tile_size = (tile_w, tile_h)
#             tile_overlap = 0.0
        else:
            tile_w, tile_h = tile_size
        tiles = __get_tiles((im_w, im_h), tile_size, tile_overlap)
        tile_ids = []

        for tile in tiles:
            img_no_str = '{:06d}'.format(img_no)
            voc_writer = Writer('{}{}{}{}'.format(slicer.ann_dst, os.sep, img_no_str, extn), tile_w, tile_h)
            for obj in objects:
                obj_lbl = obj[-4:]
                points_info = ibs.which_points_lie(obj_lbl, tile)

                if points_info == Points.NONE:
                    empty_count += 1
                    continue

                elif points_info == Points.ALL:       # All points lie inside the tile
                    new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                               obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                elif not slicer.keep_partial_labels:    # Ignore partial labels based on configuration
                    empty_count += 1
                    continue

                elif points_info == Points.P1:
                    new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                               tile_w, tile_h)

                elif points_info == Points.P2:
                    new_lbl = (0, obj_lbl[1] - tile[1],
                               obj_lbl[2] - tile[0], tile_h)

                elif points_info == Points.P3:
                    new_lbl = (obj_lbl[0] - tile[0], 0,
                               tile_w, obj_lbl[3] - tile[1])

                elif points_info == Points.P4:
                    new_lbl = (0, 0, obj_lbl[2] - tile[0],
                               obj_lbl[3] - tile[1])

                elif points_info == Points.P1_P2:
                    new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                               obj_lbl[2] - tile[0], tile_h)

                elif points_info == Points.P1_P3:
                    new_lbl = (obj_lbl[0] - tile[0], obj_lbl[1] - tile[1],
                               tile_w, obj_lbl[3] - tile[1])

                elif points_info == Points.P2_P4:
                    new_lbl = (0, obj_lbl[1] - tile[1],
                               obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                elif points_info == Points.P3_P4:
                    new_lbl = (obj_lbl[0] - tile[0], 0,
                               obj_lbl[2] - tile[0], obj_lbl[3] - tile[1])

                voc_writer.addObject(obj[0], new_lbl[0], new_lbl[1], new_lbl[2], new_lbl[3],
                                     obj[1], obj[2], obj[3])
            if slicer.ignore_empty_tiles and (empty_count == len(objects)):
                slicer._ignore_tiles.append(img_no_str)
            else:
                voc_writer.save(
                    '{}{}{}.xml'.format(slicer.ann_dst, os.sep, img_no_str))
                tile_ids.append(img_no_str)
                img_no += 1
            empty_count = 0
        mapper[im_filename] = tile_ids

    logging.info('Obtained {} annotation slices!'.format(img_no-1))
    return mapper


def unify_images(img, boxes_per_tile):
    img = np.array(img)
    first_tile = next(iter(boxes_per_tile.keys()))
    all_boxes = np.array([[]])
    all_scores = np.array([])
    all_labels = np.array([])
    
    for tile, prediction in boxes_per_tile.items():
        coords_to_add = torch.Tensor(list(map(lambda i,j: i-j, literal_eval(tile), literal_eval(first_tile))))
        boxes = torch.add(prediction['boxes'], coords_to_add, alpha=1).detach().numpy()
        all_boxes = np.append(all_boxes, boxes)
        all_scores = np.append(all_scores, prediction['scores'])
        all_labels = np.append(all_labels, prediction['labels'])
    return {"boxes": torch.from_numpy(all_boxes.reshape((-1,4))), "scores": torch.from_numpy(all_scores), "labels": torch.from_numpy(all_labels)}


def predict_tiles(img, model, is_yolo, transform, min_size=600, max_size=1333):
    tiles = slice_img_from_prediction(img, tile_size=None, tile_overlap=0.0, number_tiles=6)
    preds_image = {}
    with torch.no_grad():
        for tile in tiles:
            tile_img = img.crop(tile).convert("RGB")
            tensor_tile = transform(tile_img)
            if not is_yolo:
                predictions = model([tensor_tile])[1][0]
            else:
                predictions = model(tile_img)
                predictions = {"boxes": predictions.xyxy[0][:, :4], "scores": predictions.xyxy[0][:, 4], "labels": predictions.xyxy[0][:, 5]}
            preds_image[str(tile)] = predictions
    unified_results = unify_images(img=img, boxes_per_tile=preds_image)
    return unified_results