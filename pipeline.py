import torch
import pandas as pd 
import numpy as np
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,60).__str__() # CV_IO_MAX_IMAGE_PIXELS
os.environ["CV_IO_MAX_IMAGE_PIXELS"] = pow(2,60).__str__() 
from EmbeddingsExtractor import get_patch_embeddings
from AutoEncoder import get_reduced_patch_embeddings
from BagCreator import create_bags
from BagLevelPipeline import get_bag_model
torch.manual_seed(42)

def generate_prediction(img_file_name):  
    json_file_name = f'{img_file_name.split(".")[0]}.json'
    # Obtaining patch embeddings + valid annotation ids for a wsi 
    patch_embeddings, anns = get_patch_embeddings(img_file_name,json_file_name) 
    # Obtaining embeddings with reduced dimensions 
    reduced_patch_embeddings = get_reduced_patch_embeddings(patch_embeddings)
    bags = create_bags(reduced_patch_embeddings, anns, 50) 
    # Obtaining bag level predictions 
    predictions = [] 
    bag_model = get_bag_model()
    for bag in bags: 
        outputs_without_sigmoid, outputs_with_sigmoid = bag_model(bag) 
        predicted = (outputs_with_sigmoid >= 0.5).long().squeeze(-1) 
        predictions.append(predicted.item())
    # Obtaining final WSI-level label 

    count_0 = predictions.count(0) 
    count_1 = predictions.count(1) 

    if count_0 >= count_1: 
        final_prediction = 0 
    else: 
        final_prediction = 1 
    return final_prediction




