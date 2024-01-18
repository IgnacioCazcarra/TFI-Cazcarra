import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms as T
from sklearn.preprocessing import LabelEncoder


def get_default_transform(train=False):
    """
    Transform requerido por la clase PennFudanDataset.

    Al ser el default, no hace nada más que convertir la imagen
    a tensor para que el modelo la pueda procesar.
    """
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
    return T.Compose(transforms)


class PennFudanDataset(object):
    """
    Clase para la generación de datasets que luego toma el dataloader.

    Args:
    - csv: Dataset con columnas ['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'].
    - images_dir: directorio donde están las imagenes. Ejemplo: './data/imagenes_diagramas'
    - transforms: Función que aplica diferentes transformaciones a las imagenes. Se puede usar para
                  data augmentation y demás.
    """

    def __init__(self, csv, images_dir, transforms=get_default_transform()):
        self.csv = csv
        self.transforms = transforms
        self.images_dir = images_dir
        self.imgs = sorted(
            [
                i
                for i in os.listdir(self.images_dir)
                if os.path.join(self.images_dir, i) in self.csv["image_path"].unique()
            ]
        )

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        full_image_path = os.path.join(self.images_dir, img_path)
        img = Image.open(full_image_path).convert("RGB")
        filtered_df = self.csv[self.csv["image_path"] == full_image_path]
        # get bounding box coordinates
        num_objs = len(filtered_df)
        boxes = []
        for xmin, ymin, xmax, ymax in zip(
            filtered_df["xmin"],
            filtered_df["ymin"],
            filtered_df["xmax"],
            filtered_df["ymax"],
        ):
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.tensor(
            list(filtered_df["label_transformed"].values), dtype=torch.int64
        )

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    """
    Función collate default. Sacada de Pytorch.
    """
    return tuple(zip(*batch))


def get_dataloader(dataset, batch_size=2, shuffle=True, **kwargs):
    """
    Construye un dataloader a partir de un PennFudanDataset
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
        **kwargs
    )


def get_encoder_dict(classes_path):
    classes = pd.read_csv(classes_path)
    if "nombre" not in classes.columns:
        raise Exception(
            "ERROR: El dataframe de las clases no contiene la columna 'nombre'"
        )

    le = LabelEncoder()
    le.fit(classes.nombre)

    # Empezamos por 1 para dejarle el 0 a la 'background class'
    le_num_arr = le.transform(classes.nombre) + 1
    le_label_arr = classes.nombre.values

    le_dict = {k: v for k, v in zip(le_label_arr, le_num_arr)}
    return le_dict
