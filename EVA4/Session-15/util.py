from pathlib import Path
import os
from zipfile import ZipFile
from tqdm import tqdm
import zipfile

import torchvision
import matplotlib.pyplot as plt
import numpy as np

def extract_RAW_DATA(loc):
  archive = zipfile.ZipFile(loc)
  for file in tqdm(iterable=archive.namelist(), total=len(archive.namelist())):
    archive.extract(file, '/content/')


def extract_data(dir_list, bg_tracker):

  result = Path('/content/Unzipped_DATA/')
  result.mkdir(exist_ok=True)

  for d in dir_list:
    dir_path=f'/content/Depth_Mask_Project/{d}/'
    print(dir_path)
    
    for filename in sorted(os.listdir(dir_path)):
        if not filename.startswith('.'):
            f = dir_path+filename

            archive = zipfile.ZipFile(f)

            for file in tqdm(iterable=archive.namelist(), total=len(archive.namelist())):
              archive.extract(file, result)
              

  bg_tracker = ['Tracker_file.txt.zip', 'street_bg.zip']

  for item in bg_tracker:
    archive = zipfile.ZipFile(f'/content/drive/My Drive/{item}')
    for file in tqdm(iterable=archive.namelist(), total=len(archive.namelist())):
      
        archive.extract(file, result)

def save(tensors, name, nrow):
    try:
        tensors = tensors.detach().cpu()
    except:
        pass
    grid_tensor = torchvision.utils.make_grid(tensors, nrow = nrow )
    grid_image = grid_tensor.permute(1,2,0)
    plt.figure(figsize= (10,10))
    plt.imshow(grid_image)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(name, bbox_inches='tight')
    plt.close()



def unnormalize(img, mu, std):
  #print("unnormalized function get called. \n Shape of the Image is:{} \n Dimension of the Image is:{}".format(img.shape, img.ndimension()))
  img = img.numpy().astype(dtype=np.float32)
  
  for i in range(img.shape[0]):
    img[i] = (img[i]*std[i])+mu[i]
  
  return np.transpose(img, (1,2,0))


def unnormalize_1D(img):
  img= img.detach().cpu()
  img = img.numpy().astype(dtype=np.float32)
  img = np.transpose(img, (1,2,0))
  #print(img.shape[0])
  img = img.reshape(img.shape[0], img.shape[0])
  
  
  return img