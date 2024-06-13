# Function to pad bags with black patch embeddings 
import torch
import torch.nn as nn
import numpy as np

def get_padded_bag(bag): 

    if len(bag) > 224:
        feat_map = bag[:224]
    else:
        feat_map = []
        feat_map.extend([np.array([0.0] * 224)] * int((224 - len(bag)) / 2)) # black padding above
        feat_map.extend(bag) # feature map of patches in between
        feat_map.extend([np.array([0.0] * 224)] * int(((224 - len(bag)) / 2) + 1)) # black padding below
        feat_map = feat_map[:224]

    x = torch.tensor(np.array([feat_map])).float()
    x = nn.functional.normalize(x, dim=0, p=2)  # p=2 for L2 norm, dim=0 for cols
    bag = x 

    return torch.stack([bag]) 
# Function to create and return bags (>= 1 bag / annotation) 

def create_bags(feature_maps, anns, min_patches): 
    
    bags = [] 
    
    # Sample for dry run: 
    # anns =    [1,  1,  2,  3,  3,  3,  4,  4,  4,  4,  4,  5] 
    # patches = [1,  2,  1,  1,  2,  3,  1,  2,  3,  4,  5,  1] 
    
    i = 0 
    reserve = []
    
    # loop to traverse through each WSI 
    while (i < len(anns)): 
        
        bag = [] 
        curr_ann = anns[i] 
        
        # initial length of bag before traversing through annotation 
        prev_len = len(bags) 
        
        # loop to traverse through each annotation 
        while (i < len(anns) and curr_ann == anns[i]): 
            curr_ann = anns[i] 
            bag.append(feature_maps[i]) 
            i += 1 
        
        # minimum number of patches in a bag to keep it 
        if len(bag) >= min_patches: 
            # maximum number of patches in a bag before splitting it 
            if len(bag) < 224: 
                bags.append(get_padded_bag(bag)) 
            
            else: 
                x = 0 
                while (x + 224) < len(bag): 
                    bags.append(get_padded_bag(bag[x : x + 224])) 
                    x += 224 
        
        # len(bag) < min patches 
        else: 
            reserve.extend(bag) 
        
        # new length of bag after traversing through annotation 
        new_len = len(bags) 
    
    # size of extra bags (can be modified) 
    extra_bags_size = int(min_patches * 1.5) 

    x = 0 
    # loop to off load reserve once done with a given wsi 
    while (x + extra_bags_size) <= len(reserve): 
        bags.append(get_padded_bag(reserve[x : x + extra_bags_size])) 
        x += extra_bags_size 
    
    return bags 
