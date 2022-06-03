import torch
import clip
from PIL import Image
import argparse
import os
import torch
from torchvision import datasets, transforms
import numpy as np
import sys
from scipy import linalg
from tqdm import tqdm

def calc_fcd_score(batch_feature_source_np, batch_feature_test_np):
    mu1 = np.mean(batch_feature_source_np, axis=0)
    sigma1 = np.cov(batch_feature_source_np, rowvar=False)
    mu2 = np.mean(batch_feature_test_np, axis=0)
    sigma2 = np.cov(batch_feature_test_np, rowvar=False)
    curr_fcd = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return curr_fcd

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    # Numpy implementation of the Frechet Distance.
    # The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    # and X_2 ~ N(mu_2, C_2) is
    # d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    # Stable version by Dougal J. Sutherland.
    
    
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

    
def from_imgs_folder_to_process_tensor(path, preprocess, sourceFlag = False):
    arr = []
    print("Preprocess Images from : " + path)
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in tqdm(filenames):
            if filename.endswith('.png'):
                curr_path = dirpath + "/" + filename
                curr_img = preprocess(Image.open(curr_path))
                arr.append(curr_img)
    imgs_tensor = torch.stack(arr)
    if sourceFlag:
        torch.save(imgs_tensor, 'orig_data_base.pt')
    return imgs_tensor
    
    
parser = argparse.ArgumentParser()
parser.add_argument('-p1', '--path_source', type=str, default="/home/source/")
parser.add_argument('-p2', '--path_test', type=str, default="/home/test/")
parser.add_argument('-bs', '--batch_size', type=int, default=32)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
batch_size = args.batch_size
fname = "results_fcd.txt"

tensor_source = from_imgs_folder_to_process_tensor(args.path_source, preprocess, sourceFlag = True)
tensor_test = from_imgs_folder_to_process_tensor(args.path_test, preprocess)
tensor_source = tensor_source[:tensor_test.shape[0],:,:,:]
    
dl_source = torch.utils.data.DataLoader(tensor_source, batch_size=batch_size,shuffle=False)
dl_test = torch.utils.data.DataLoader(tensor_test, batch_size=batch_size,shuffle=False)    

it_test = iter(dl_test)
fcd_res = []
feature_tensor_source_arr = []
feature_tensor_test_arr = []

print("Infernce from CLIP\n")                         
for i, batch_source in tqdm(enumerate(dl_source)):
    batch_test = next(it_test)
    with torch.no_grad():
        batch_feature_tensor_source = model.encode_image(batch_source.to(device))
        batch_feature_tensor_test = model.encode_image(batch_test.to(device))

    feature_tensor_source_arr.append(batch_feature_tensor_source.cpu().numpy())
    feature_tensor_test_arr.append(batch_feature_tensor_test.cpu().numpy())
    
feature_np_test = np.vstack(feature_tensor_test_arr)
feature_np_source = np.vstack(feature_tensor_source_arr)

print("Calc FCD Score:\n") 
res = calc_fcd_score(feature_np_source, feature_np_test)
np.savetxt(fname, [res])
print(res)
