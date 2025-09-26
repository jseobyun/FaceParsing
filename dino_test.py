import os
import cv2
import random
import numpy as np
import torch
from src.models.encoder import Encoder
from sklearn.decomposition import PCA

def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(384, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # H W 3
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).cuda().permute(2, 0 ,1)
    img = img.unsqueeze(dim=0)

    return img 

def process_feature(features):
    features = features[0].detach().cpu().permute(1,2,0) # 32 32 1024
    features = features.numpy()
    features = features.reshape(-1, 1024)
    return features

def do_pca(features_list):

    features = np.concatenate(features_list, axis=0)
    num_imgs = len(features_list)
    H, W = 32, 24
    pca = PCA(n_components=3)
    # reduced = pca.fit_transform(features)
    pca.fit(features[:H*W])
    reduced = pca.transform(features)[:, :3]

    # 0-1로 정규화
    reduced_min = reduced.min(axis=0)
    reduced_max = reduced.max(axis=0)
    normalized = (reduced - reduced_min) / (reduced_max - reduced_min)
    
    pca_vis = []
    for i in range(num_imgs):
        pca_vis.append(normalized[i*H*W:(i+1)*H*W, :].reshape(H, W, 3))
        
    return pca_vis
    
if __name__ == "__main__":
    encoder = Encoder()

    img_dir = "/home/jseob/Downloads/TEST/hands/images"
    save_dir = "/home/jseob/Downloads/TEST/hands/results"
    os.makedirs(save_dir, exist_ok=True)

    file_names = sorted(os.listdir(img_dir))

    feature_list = []
    for file_name in file_names:
        img_path = os.path.join(img_dir, file_name)
        img = load_img(img_path)
       

        img_features = encoder.extract_dino_feature(img)        
        img_features = process_feature(img_features)
        feature_list.append(img_features)        

    pca_vis = do_pca(feature_list)        
        
    canvas = np.concatenate(pca_vis, axis=1)
    canvas = (255*canvas).astype(np.uint8)
    canvas = cv2.resize(canvas, dsize=(0,0), fx=4, fy=4)
        
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, canvas)
        
        

