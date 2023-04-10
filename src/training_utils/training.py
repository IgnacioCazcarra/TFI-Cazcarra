import logging
import torch
import torchvision
import numpy as np
from .dataset import PennFudanDataset
from ..constants import *

logging.basicConfig(level = logging.INFO)

try:
    from ..detection.engine import train_one_epoch, evaluate
except Exception as e:
    logging.error("Could not import functions from ..detection.engine file. If you're predicting, ignore this.")



def get_model_instance_segmentation(num_classes, model_type, **kwargs):
    '''
    Devuelve una instancia de modelo según se especifique.
    
    Args:
    - num_classes: Numero de clases a predecir.
    - model_type: Arquitectura de detección de objetos a utilizar. Opciones -> ['RETINANET','FASTER-RCNN']
    '''
    if model_type.upper() == "RETINANET":
        model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes, pretrained=True, **kwargs)
    elif model_type.upper() == "FASTER-RCNN":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes, pretrained=True, **kwargs)
    else:
        raise Exception("Arquitectura no soportada. Opciones disponibles: ['RETINANET','FASTER-RCNN']")
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nbr_params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f"Instancing model {model_type}. Trainable parameters: {nbr_params}")
    
    return model


def get_default_optimizer(params):
    return torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


def get_default_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    

def train_model(model, data_loader, data_loader_test, params, device, num_epochs=30, 
                optimizer=None, lr_scheduler=None, override_path_final=None, override_path_best=None):
    save_best_model = SaveBestModel()
    if not optimizer:
        optimizer = get_default_optimizer(params)
    if not lr_scheduler:
        lr_scheduler = get_default_lr_scheduler(optimizer)
    
    for epoch in range(num_epochs):
        metric_logger, loss_value = train_one_epoch(model, optimizer, data_loader, 
                                                    device, epoch, print_freq=10)
        del metric_logger # Para no ocupar espacio, por las dudas
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
        save_best_model(loss_value, epoch, model, override_path=override_path_best)
    
    model_name = model.__class__.__name__.lower()
    if not override_path_final:
        override_path_final = os.path.join(PATH, "data", "all_models", f"model_{model_name}_final.pt")
    save_model(path_to_save=override_path_final, model=model, epoch=epoch, 
               loss_value=loss_value)

    
def save_model(path_to_save, model, epoch, loss_value):
    '''
    Guarda el modelo
    '''
    logging.info("Guardando...")
    model_scripted = torch.jit.script(model)
    model_scripted.save(path_to_save)
    logging.info(f"Modelo guardado en {path_to_save}")
    
    
def load_model(path_to_load):
    '''
    Carga el modelo
    '''
    return torch.jit.load(path_to_load)


class SaveBestModel:
    """
    Va guardando el mejor modelo en cada epoch basandose en el valor de la Loss Function.
    """
    def __init__(self, best_loss=float('inf')):
        self.best_loss = best_loss
        
    def __call__(self, current_loss, epoch, model, override_path=None):
        if current_loss < self.best_loss:
            self.best_loss = round(current_loss,3)
            logging.info(f"\nBest loss: {self.best_loss}")
            logging.info(f"Saving best model for epoch: {epoch+1}\n")
            
            if not override_path:
                model_name = model.__class__.__name__.lower()
                PATH_TO_SAVE_MODEL = os.path.join(PATH, "data", "models", f"best_model_{model_name}.pt")
            else:
                PATH_TO_SAVE_MODEL = override_path
                
            save_model(path_to_save=PATH_TO_SAVE_MODEL, model=model, 
                       epoch=epoch, loss_value=self.best_loss)