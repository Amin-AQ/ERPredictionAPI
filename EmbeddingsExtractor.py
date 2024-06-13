import numpy as np
import os
import cv2
import json 
import torch
import torchvision
from FeatureExtractor import get_feature_extractor

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),  # resize to 224*224
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # normalization
    ]
)

def get_patch_feature_embedding(img,patch_feature_extractor): 
    img = torch.from_numpy(img.astype(np.double)) 
    img = img.permute(2, 0, 1) 
    img = transform(img) 
    img = img.float() 
    img = torch.unsqueeze(img, dim=0) 
    return patch_feature_extractor(img) 


# Function to slide over annotated regions of a wsi forming patches. Returns embeddings of patches after passing through specialized resnet50 + valid annotation ids 

def get_patch_embeddings(image_file_name, json_file_name): 
    patch_feature_extractor = get_feature_extractor()
    annotation_ids = [] 
    patch_embeddings = [] 
    
    # Construct the absolute path to the image and JSON files
    image_path =  f'./images/{image_file_name}'
    json_path = f'./jsons/{json_file_name}'
    print(image_path)
    # Load the image path 
    img = cv2.imread(image_path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    # Load the JSON file containing coordinates
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Get the original image dimensions
    height, width = img.shape[:2] 
    
    i = 0
    for annotation in data['positive']:
        i = i + 1
        vertices = annotation['vertices']
        vertices = np.array(vertices) 
        vertices = vertices.astype(int) 
        max_x, max_y = vertices.max(0) 
        min_x, min_y = vertices.min(0) 
        
        min_x = int(min_x - (0.05 * (max_x - min_x))) 
        max_x = int(max_x + (0.05 * (max_x - min_x))) 
        min_y = int(min_y - (0.05 * (max_y - min_y))) 
        max_y = int(max_y + (0.05 * (max_y - min_y))) 
        
        # Check if the image was loaded successfully
        if img is None:
            print("Error: Unable to load the image.")
        else:
            # Check if the region of interest is within the bounds of the image
            if min_x >= 0 and min_y >= 0 and max_x <= img.shape[1] and max_y <= img.shape[0]:
                # Extract the rectangular patch from the image
                roi = img[min_y:max_y, min_x:max_x]

                # Check if the extracted roi is valid and not empty
                if not roi.size == 0:
                    try: 
                        # extracting patches from the region of interest 
                        y = min_y 
                        j = 0 
                        
                        # sliding vertically 
                        while (y < max_y): 
                            # sliding horizontally 
                            x = min_x 
                            while (x < max_x): 
                                j = j + 1 
                                
                                # obtaining next patch 
                                next_patch = img[y:y+224, x:x+224] 
                                
                                # storing annotation IDs 
                                annotation_ids.append(i) 
                                
                                # storing feature vector 
                                embedding = get_patch_feature_embedding(next_patch,patch_feature_extractor) 
                                squeezed_arr = np.squeeze(embedding) 
                                patch_embeddings.append(squeezed_arr) 
                                
                                # sliding horizontally 
                                x = x + 224 
                            
                            # sliding vertically 
                            y = y + 224 
                            
                    except cv2.error as e:
                        print(f"Error during resizing: {str(e)}")
                else:
                    print("Error: The extracted region of interest (roi) is empty.")
            else:
                print("Error: Region of interest coordinates are out of bounds.")
    
    return patch_embeddings, annotation_ids 
