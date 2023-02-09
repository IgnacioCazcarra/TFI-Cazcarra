import sys
sys.path.append("../")

from src.constants import *
from src.training_utils.dataset import *
from src.training_utils.training import train_model, get_model_instance_segmentation

import torch
import pandas as pd

from torchvision import transforms as T


def get_custom_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.RandomEqualize(0.5))
        transforms.append(T.RandomInvert(0.3))
        transforms.append(T.RandomGrayscale(0.2))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

train_df = pd.read_csv("/home/nacho/TFI-Cazcarra/data/csv/augmented_train_diagramas.csv", header=None)
train_df.columns = ['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
test_df = pd.read_csv("/home/nacho/TFI-Cazcarra/data/csv/augmented_test_diagramas.csv")
train_df = train_df[train_df['label']=="tabla"]
test_df = test_df[test_df['label']=="tabla"]

IMAGES_DIR = f"{PATH}/data/imagenes_diagramas/"

le_dict = get_encoder_dict(CLASSES_CSV)

train_df['label_transformed'] = train_df['label'].apply(lambda x: le_dict[x])
test_df['label_transformed'] = test_df['label'].apply(lambda x: le_dict[x])

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2 # Tablas + background

dataset = PennFudanDataset(csv=train_df, images_dir=IMAGES_DIR, transforms=get_custom_transform(train=True))
dataset_test = PennFudanDataset(csv=test_df, images_dir=IMAGES_DIR, transforms=get_custom_transform(train=False))

data_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
data_loader_test = get_dataloader(dataset_test, batch_size=1, shuffle=False)

train = True
epochs = 20

model = get_model_instance_segmentation(num_classes=num_classes, model_type="retinanet", min_size=600)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

init_lr = 0.00005 
weight_decay = init_lr * 100 
optimizer = torch.optim.AdamW(params, lr=init_lr, weight_decay=weight_decay)

override_path_best = f"{PATH}/data/models/model_best_tablas_retinanet.pt"
override_path_final = f"{PATH}/data/models/model_final_tablas_retinanet.pt"

if train:
    train_model(model=model, data_loader=data_loader, data_loader_test=data_loader_test, 
            num_epochs=epochs, device=device, params=params, optimizer=optimizer,
            override_path_best=override_path_best, override_path_final=override_path_final)
