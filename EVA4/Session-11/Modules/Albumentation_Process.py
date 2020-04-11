import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor

class Albumentations:
  def __init__(self,Normalize_mean_std=None,Rotate=None,Padding = None, R_Crop = None, cutout=None):
      
      print("in the init of Albumentation" )
      
      self.transforms=[]
      if Rotate is not None:
          self.transforms.append(A.Rotate(Rotate))
      if Padding is not None:
          self.transforms.append(A.PadIfNeeded(min_height=40, min_width=40, always_apply=True))
      if R_Crop is not None:
          self.transforms.append(A.RandomCrop(32,32, always_apply=True))
      
      if Normalize_mean_std is not None:
          self.transforms.append(A.Normalize(Normalize_mean_std[0],Normalize_mean_std[1], always_apply=True))
      if cutout is not None:
          self.transforms.append(A.Cutout(*cutout))
      self.transforms.append(ToTensor())
      
      print("Finally Append become:{}".format(self.transforms))
      self.Transforms=A.Compose(self.transforms)
      
  def __call__(self,img):
      
      #print("__call__ is called in Albumentation")
      
      img = np.fliplr(img)
      
      img=np.array(img)
      #img = np.fliplr(img)
      
      #img=self.Transforms(image=img)
      
   #   print("value of img without ['image']is: {}".format(img))

    
      img=self.Transforms(image=img)['image']
      #print("value of img is: {}".format(img))
      return img