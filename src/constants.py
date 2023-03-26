ELEMENT_TO_TRAIN = "diagramas"
PATH = "/home/nacho/TFI-Cazcarra"

IMAGES_DIR = f"{PATH}/data/imagenes_diagramas"
CLASSES_CSV = f"{PATH}/data/csv/classes_{ELEMENT_TO_TRAIN}.csv"

le_dict_cardinalidades = {'muchos_opcional': 2,
                          'muchos_obligatorio': 1,
                          'uno_opcional': 3,
                          'uno_obligatorio': 4}