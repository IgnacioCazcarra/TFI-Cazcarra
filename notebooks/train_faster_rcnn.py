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
    transforms.append(T.ToTensor())
    if train:
#        pass
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)

#train_df = pd.read_csv("/home/nacho/TFI-Cazcarra/data/csv/augmented_train_diagramas.csv", header=None)
#train_df.columns = ['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
#test_df = pd.read_csv("/home/nacho/TFI-Cazcarra/data/csv/augmented_test_diagramas.csv")

train_df = pd.read_csv(f"/home/nacho/TFI-Cazcarra/data/tiles/train_cardinalidades_linux_fixed.csv")
test_df = pd.read_csv(f"/home/nacho/TFI-Cazcarra/data/tiles/test_cardinalidades_linux_fixed.csv")
IMAGES_DIR = f"{PATH}/data/tiles/image_slices/"

le_dict = get_encoder_dict(CLASSES_CSV)

train_df['label_transformed'] = train_df['label'].apply(lambda x: le_dict[x])
test_df['label_transformed'] = test_df['label'].apply(lambda x: le_dict[x])

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = len(le_dict)+1 

dataset = PennFudanDataset(csv=train_df, images_dir=IMAGES_DIR, transforms=get_custom_transform(train=True))
dataset_test = PennFudanDataset(csv=test_df, images_dir=IMAGES_DIR, transforms=get_custom_transform(train=False))

data_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
data_loader_test = get_dataloader(dataset_test, batch_size=1, shuffle=False)

train = True
epochs = 50

model = get_model_instance_segmentation(num_classes=num_classes, model_type="faster-rcnn", min_size=700)

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead

anchor_generator = AnchorGenerator(
    sizes=tuple([(8, 16, 32, 64, 128) for _ in range(5)]),
    aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0, 3.0) for _ in range(5)]))

model.rpn.anchor_generator = anchor_generator
model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

init_lr = 0.00005 
weight_decay = init_lr * 100 
optimizer = torch.optim.AdamW(params, lr=init_lr, weight_decay=weight_decay)

override_path = f"{PATH}/data/models/model_best_cardinalidades_fasterrcnn.pt"
if train:
    train_model(model=model, data_loader=data_loader, data_loader_test=data_loader_test, 
            num_epochs=epochs, device=device, params=params, optimizer=optimizer,
            override_path=override_path)
