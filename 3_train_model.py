from ultralytics import YOLO
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import cv2

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

if __name__ == '__main__':
    #torch.cuda.empty_cache()
    free_gpu_cache()
    
    # Load a model
    model = YOLO('yolov10n.pt')  # Load pretrained model

    # Train the model
    model.train(data="dataset.yaml", epochs=50, batch=16, imgsz=640, pretrained=True, single_cls=False, patience=5, dropout=0.1, verbose=True, device=0, save_period=2)

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category

    print(model.names)
