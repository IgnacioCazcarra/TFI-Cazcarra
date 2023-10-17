import os

ELEMENT_TO_TRAIN = "diagramas"
# We go a level above src/ to find the current root dir.
PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

IMAGES_DIR = os.path.join(PATH, "data", "imagenes_diagramas")
CLASSES_CSV = os.path.join(PATH, "data", "csv", f"classes_{ELEMENT_TO_TRAIN}.csv")

le_dict_cardinalidades = {
    "muchos_opcional": 2,
    "muchos_obligatorio": 1,
    "uno_opcional": 3,
    "uno_obligatorio": 4,
}
