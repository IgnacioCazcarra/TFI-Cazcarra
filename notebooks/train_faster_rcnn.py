import sys
sys.path.append("../")

from src.constants import *
from src.training_utils.dataset import *
from src.training_utils.training import train_model, get_model_instance_segmentation

import torch
import pandas as pd

from torchvision import transforms as T
from torch.utils.data import WeightedRandomSampler


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

# train_df = pd.read_csv(f"/home/nacho/TFI-Cazcarra/data/tiles/train_cardinalidades_linux_fixed.csv")
# test_df = pd.read_csv(f"/home/nacho/TFI-Cazcarra/data/tiles/test_cardinalidades_linux_fixed.csv")
# IMAGES_DIR = f"{PATH}/data/tiles/image_slices/"

le_dict = get_encoder_dict(CLASSES_CSV)

train_df['label_transformed'] = train_df['label'].apply(lambda x: le_dict[x])
test_df['label_transformed'] = test_df['label'].apply(lambda x: le_dict[x])

# train_df['class_weight'] = train_df['label'].apply(lambda x: 0.5 if "opcional" in x else 0.25)
# class_weights = []
# for path in sorted(train_df['image_path'].unique()):
#     filtered = train_df[train_df['image_path']==path]
#     class_weight = max(filtered['class_weight'].values)
#     class_weights.append(class_weight)

# sampler = WeightedRandomSampler(weights=class_weights, num_samples=2*len(train_df['image_path'].unique()), 
#                                 replacement=True)
    
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = len(le_dict)+1 

dataset = PennFudanDataset(csv=train_df, images_dir=IMAGES_DIR, transforms=get_custom_transform(train=True))
dataset_test = PennFudanDataset(csv=test_df, images_dir=IMAGES_DIR, transforms=get_custom_transform(train=False))

data_loader = get_dataloader(dataset, batch_size=2, shuffle=True)#, sampler=sampler)
data_loader_test = get_dataloader(dataset_test, batch_size=1, shuffle=False)

train = True
epochs = 50

model = get_model_instance_segmentation(num_classes=num_classes, model_type="faster-rcnn", min_size=600)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

init_lr = 0.00005 
weight_decay = init_lr * 100 
optimizer = torch.optim.AdamW(params, lr=init_lr, weight_decay=weight_decay)

override_path_best = f"{PATH}/data/models/model_best_cardinalidades_fasterrcnn.pt"
override_path_final = f"{PATH}/data/models/model_final_cardinalidades_fasterrcnn.pt"

if train:
    train_model(model=model, data_loader=data_loader, data_loader_test=data_loader_test, 
            num_epochs=epochs, device=device, params=params, optimizer=optimizer,
            override_path_best=override_path_best, override_path_final=override_path_final)
