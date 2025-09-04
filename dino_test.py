import os
import cv2
import random
import numpy as np
import torch
from src.models.encoder import Encoder
from sklearn.decomposition import PCA

def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(1024, 1024))
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
    H, W = 64, 64
    pca = PCA(n_components=3)
    # reduced = pca.fit_transform(features)
    pca.fit(features[:H*W])
    reduced = pca.transform(features)[:, :3]

    # 0-1로 정규화
    reduced_min = reduced.min(axis=0)
    reduced_max = reduced.max(axis=0)
    normalized = (reduced - reduced_min) / (reduced_max - reduced_min)
    
    
    pca_vis = [
        normalized[:H*W, :].reshape(H, W, 3),
        normalized[H*W:2*H*W, :].reshape(H, W, 3),
        normalized[-H*W:, :].reshape(H, W, 3),
    ]
    return pca_vis
    
if __name__ == "__main__":
    encoder = Encoder()

    img_dir = "/home/jseob/Downloads/TEST/0027/e0_results/frames/000"
    save_dir = "/home/jseob/Downloads/TEST/0027/e0_results/frames/000_pca"
    os.makedirs(save_dir, exist_ok=True)

    file_names = sorted(os.listdir(img_dir))

    img_ids = [file_name.split("_")[0] for file_name in file_names]
    random.shuffle(img_ids)
    

    for idx, img_id in enumerate(img_ids):
        rgb_path = os.path.join(img_dir.replace("0027", "0028"), img_ids[idx]+"_rgb.jpg")
        depth_path = os.path.join(img_dir, img_ids[idx+1]+"_depth.jpg")
        normal_path = os.path.join(img_dir, img_ids[idx+2]+"_normal.jpg")
        # depth_path = os.path.join(img_dir, img_id+"_depth.jpg")
        # normal_path = os.path.join(img_dir, img_id+"_normal.jpg")
    
        rgb = load_img(rgb_path)
        depth = load_img(depth_path)
        normal = load_img(normal_path)    

        rgb_features = encoder.extract_dino_feature(rgb)
        depth_features = encoder.extract_dino_feature(depth)
        normal_features = encoder.extract_dino_feature(normal)


        rgb_features = process_feature(rgb_features)
        depth_features = process_feature(depth_features)
        normal_features = process_feature(normal_features)
        

        pca_vis = do_pca([rgb_features, depth_features, normal_features])        

        rgb_cv = cv2.imread(rgb_path)
        depth_cv = cv2.imread(depth_path)
        normal_cv = cv2.imread(normal_path)

        rgb_cv = cv2.resize(rgb_cv, dsize=(256, 256))
        depth_cv = cv2.resize(depth_cv, dsize=(256, 256))
        normal_cv = cv2.resize(normal_cv, dsize=(256, 256))
        input = np.concatenate([rgb_cv, depth_cv, normal_cv], axis=1)
        
        canvas = np.concatenate(pca_vis, axis=1)
        canvas = (255*canvas).astype(np.uint8)
        canvas = cv2.resize(canvas, dsize=(0,0), fx=4, fy=4)
        canvas = np.concatenate([input, canvas], axis=0)
        
        save_path = os.path.join(save_dir, img_id+".jpg")
        cv2.imwrite(save_path, canvas)
        
        


