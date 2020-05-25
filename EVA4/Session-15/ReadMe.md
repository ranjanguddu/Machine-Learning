# Project

This project is about the Semantic Segmentation of the image  as well as creating the depth map of the same. To prepare this Model we have used 400K images with their mask and depth image. So, in total we vae 1200K images to deal with. To know more about the used dataset please refer [here](https://github.com/ranjanguddu/EVA4-Session-14)

---
## Input  to the Model
---
BG and FG_BG Images are 3 chaneel images where as the mask and the depth images are 1 gray-scal image. Below is the preview: 

![input]<img src="https://github.com/ranjanguddu/Machine-Learning/blob/master/EVA4/Session-15/Input_fig.png" width="100">

## On-boarding the data in Colab:
___

There are 400K images of FG_BG, and corresponding Mask and Depth Images.
So in total we need to deal with 1200K images.

To deal with huge data I have divided the entire dataset in 10 batches having each of size 40K images. 

Then uploadindg the data in My Drive and then unzipping the entire folder in Colab RAM.




```python
def extract_RAW_DATA(loc):
  archive = zipfile.ZipFile(loc)
  for file in tqdm(iterable=archive.namelist(), total=len(archive.namelist())):
    archive.extract(file, '/content/')
```

After running the above script, the folder structure look like:
![form-1](https://github.com/ranjanguddu/Machine-Learning/blob/master/EVA4/Session-15/fig-1.png)


Then we unzip all the zip files present in those folder in a folder name \'Unzipped_DATA'. Then run the below script:

```python

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
```

Now the folder structure look like:
![form-2](https://github.com/ranjanguddu/Machine-Learning/blob/master/EVA4/Session-15/fig-2.png)

## Prepare Dataset 
---

```python
class DepthMaskDataset(Dataset):
    def __init__(self, root, image_size, num_data, transform=True):
        
        csv_file = root/'Tracker_file.txt'
        self.annotations = pd.read_csv(csv_file, sep=',', header = None, nrows=num_data)
        self.annotations_shuffle = self.annotations.sample(frac=1)
        self.root = root
        self.imsize = image_size
        
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        N = 40000
        
      
        self.image_folder = self.root/f'image_batch{index//N+1}'
        self.mask_folder = self.root/f'mask_batch{index//N+1}'
        self.depth_folder = self.root/f'depth_image_batch{index//N+1}'
        self.bg_folder = self.root/'street_bg'

        image_path = os.path.join(self.image_folder, self.annotations.iloc[index, 0])
        
        image = Image.open(image_path)
        
        bg_path = os.path.join(self.bg_folder, self.annotations.iloc[index, 1])
        bg = Image.open(bg_path)
        
        mask_path = os.path.join(self.mask_folder, self.annotations.iloc[index, 2])
        mask = Image.open(mask_path)
        
        depth_path = os.path.join(self.depth_folder, self.annotations.iloc[index, 3])
        depth = Image.open(depth_path)

        

        if self.transform:

            bg_mean = [0.4066298566301917, 0.4002413496649073, 0.39249680184578123]
            bg_std = [0.2516888771733121, 0.25270893840650427, 0.2627707634116584]

            img_mean = [0.406589950038055, 0.3974733626335778, 0.3922238309037007]
            img_std = [0.2537575580108365, 0.2544099256403426, 0.26358312983854404]


            size = self.imsize

            bg_transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(bg_mean, bg_std)])
            img_transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(img_mean, img_std)])
            mask_transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), ])
            depth_transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), ])

            
                

            
            bg = bg_transform(bg)
            image= img_transform(image)
            mask = mask_transform(mask)
            depth = depth_transform(depth)
        return(bg, image, mask, depth)

```
## Prepare Dataloader:
```python

root_dir='/content/Unzipped_DATA'

dataset = DepthMaskDataset(root = root_dir, image_size=128, num_data=400000)  # transform=transforms.ToTensor()
train_set, test_set = torch.utils.data.random_split(dataset, [280000,120000])
```

## Check the data coming from dataloader:
```python
train_loader = DataLoader(dataset=train_set, batch_size = 16, shuffle=True)
dataiter = iter(train_loader)
data = dataiter.next()

for i in range(4):
    print(data[i].shape)  
```
The output will be:
```
torch.Size([16, 3, 224, 224])
torch.Size([16, 3, 224, 224])
torch.Size([16, 1, 224, 224])
torch.Size([16, 1, 224, 224])
```

## Creating the Model:
---
The best Model which suits to create image segmenatation is UNET. So, I tried with various model based in UNET but the best result I have acheived is mentioned below:
```python
class Resnet_v1(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Resnet_v1, self).__init__()
        self.conv1_k3 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2_k1 = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=False)

        self.conv3_k3 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_k1 = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.conv5_k1 = nn.Conv2d(input_channel,output_channel, kernel_size=1, stride=2, padding=0, bias=False)


    def forward(self, x):
        same = self.conv5_k1(x)
        out = self.conv1_k3(x)
        out = self.conv2_k1(out)
        out = self.bn1(out)
        out = self.relu(out)
        #out = nn.ReLU(nn.BatchNorm3d(self.conv2_k1(self.conv1_k3(x))), inplace=False)

        out = self.conv3_k3(out)
        out = self.conv4_k1(out)
        out = self.bn2(out)

        out += same

        out = self.relu(out)

        return out

class Resnet_v2(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Resnet_v2, self).__init__()
        self.conv1_k3 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_k1 = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=False)

        self.conv3_k3 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_k1 = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.conv5_k3 = nn.ConvTranspose2d(input_channel,output_channel, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self,x):
        same = self.conv5_k3(x)
        out = self.conv1_k3(same)
        out = self.conv2_k1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3_k3(out)
        out = self.conv4_k1(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out

class UNET_Model(nn.Module):
    def __init__(self, n_class):
        super(UNET_Model, self).__init__()
        self.initconv = nn.Sequential(
            nn.Conv2d(6,64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rb1 = Resnet_v1(64,128)
        self.rb2 = Resnet_v1(128,256)
        self.rb3 = Resnet_v1(256, 512)
        
        self.rb4 = Resnet_v2(512, 256)
        self.rb5 = Resnet_v2(256, 128)
        self.rb6 = Resnet_v2(128, 64)
        self.rb7 = Resnet_v2(64, 64)
        self.rb8 = Resnet_v2(64, 32)

        self.lastconv = nn.Sequential(nn.Conv2d(32,n_class,kernel_size=1, stride=1, padding=0, bias=False))

        self.rb4d = Resnet_v2(512, 256)
        self.rb5d = Resnet_v2(256, 128)
        self.rb6d = Resnet_v2(128, 64)
        self.rb7d = Resnet_v2(64, 64)
        self.rb8d = Resnet_v2(64, 32)

        self.lastconvD = nn.Sequential(nn.Conv2d(32,n_class,kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, data):

        out0 = self.initconv(data)
        out1 = self.maxpool(out0)
        out2 = self.rb1(out1)
        out3 = self.rb2(out2)
        out4 = self.rb3(out3)

        out4 = nn.functional.interpolate(out4, scale_factor=2, mode='bilinear')

        out5 = self.rb4(out4)
        out5 += out3

        out5 = nn.functional.interpolate(out5, scale_factor=2, mode='bilinear')

        out6 = self.rb5(out5)
        out6 += out2

        out6 = nn.functional.interpolate(out6, scale_factor=2, mode='bilinear')

        out7 = self.rb6(out6)
        out7 += out1

        out7 = nn.functional.interpolate(out7, scale_factor=2, mode='bilinear')

        out8 = self.rb7(out7)
        out8 += out0

        out9 = nn.functional.interpolate(out8, scale_factor=2, mode='bilinear')
        out9 = self.rb8(out9)

        mask_out = self.lastconv(out9)

        ############  for Depth prediction ####################

        out5D = self.rb4d(out4)
        out5D = nn.functional.interpolate(out5D, scale_factor=2, mode='bilinear') 

        out6D = self.rb5d(out5D)
        out6D += out2

        out6D = nn.functional.interpolate(out6D, scale_factor=2, mode='bilinear')
        out7D = self.rb6d(out6D)

        out7D += out1

        out7D = nn.functional.interpolate(out7D, scale_factor=2, mode='bilinear')

        out8D = self.rb7d(out7D)
        out8D  += out0

        out9D = nn.functional.interpolate(out8D, scale_factor=2, mode='bilinear')
        out9D = self.rb8d(out9D)

        depth_out = self.lastconvD(out9D)
        
        return mask_out, depth_out
```
Above model is comletey inspired by my instructor, [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/?originalSubdomain=in)

## Summary of the Model:
```python
from torchsummary import summary

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = UNET_Model(n_class=1).to(device)
summary(model, input_size=torch.cat((data[0][0], data[1][0])).shape)

```

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]          18,816
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
         MaxPool2d-4           [-1, 64, 56, 56]               0
            Conv2d-5          [-1, 128, 28, 28]           8,192
            Conv2d-6          [-1, 128, 28, 28]          73,728
            Conv2d-7          [-1, 128, 28, 28]          16,384
       BatchNorm2d-8          [-1, 128, 28, 28]             256
              ReLU-9          [-1, 128, 28, 28]               0
           Conv2d-10          [-1, 128, 28, 28]         147,456
           Conv2d-11          [-1, 128, 28, 28]          16,384
      BatchNorm2d-12          [-1, 128, 28, 28]             256
             ReLU-13          [-1, 128, 28, 28]               0
        Resnet_v1-14          [-1, 128, 28, 28]               0
           Conv2d-15          [-1, 256, 14, 14]          32,768
           Conv2d-16          [-1, 256, 14, 14]         294,912
           Conv2d-17          [-1, 256, 14, 14]          65,536
      BatchNorm2d-18          [-1, 256, 14, 14]             512
             ReLU-19          [-1, 256, 14, 14]               0
           Conv2d-20          [-1, 256, 14, 14]         589,824
           Conv2d-21          [-1, 256, 14, 14]          65,536
      BatchNorm2d-22          [-1, 256, 14, 14]             512
             ReLU-23          [-1, 256, 14, 14]               0
        Resnet_v1-24          [-1, 256, 14, 14]               0
           Conv2d-25            [-1, 512, 7, 7]         131,072
           Conv2d-26            [-1, 512, 7, 7]       1,179,648
           Conv2d-27            [-1, 512, 7, 7]         262,144
      BatchNorm2d-28            [-1, 512, 7, 7]           1,024
             ReLU-29            [-1, 512, 7, 7]               0
           Conv2d-30            [-1, 512, 7, 7]       2,359,296
           Conv2d-31            [-1, 512, 7, 7]         262,144
      BatchNorm2d-32            [-1, 512, 7, 7]           1,024
             ReLU-33            [-1, 512, 7, 7]               0
        Resnet_v1-34            [-1, 512, 7, 7]               0
  ConvTranspose2d-35          [-1, 256, 14, 14]       1,179,648
           Conv2d-36          [-1, 256, 14, 14]         589,824
           Conv2d-37          [-1, 256, 14, 14]          65,536
      BatchNorm2d-38          [-1, 256, 14, 14]             512
             ReLU-39          [-1, 256, 14, 14]               0
           Conv2d-40          [-1, 256, 14, 14]         589,824
           Conv2d-41          [-1, 256, 14, 14]          65,536
      BatchNorm2d-42          [-1, 256, 14, 14]             512
             ReLU-43          [-1, 256, 14, 14]               0
        Resnet_v2-44          [-1, 256, 14, 14]               0
  ConvTranspose2d-45          [-1, 128, 28, 28]         294,912
           Conv2d-46          [-1, 128, 28, 28]         147,456
           Conv2d-47          [-1, 128, 28, 28]          16,384
      BatchNorm2d-48          [-1, 128, 28, 28]             256
             ReLU-49          [-1, 128, 28, 28]               0
           Conv2d-50          [-1, 128, 28, 28]         147,456
           Conv2d-51          [-1, 128, 28, 28]          16,384
      BatchNorm2d-52          [-1, 128, 28, 28]             256
             ReLU-53          [-1, 128, 28, 28]               0
        Resnet_v2-54          [-1, 128, 28, 28]               0
  ConvTranspose2d-55           [-1, 64, 56, 56]          73,728
           Conv2d-56           [-1, 64, 56, 56]          36,864
           Conv2d-57           [-1, 64, 56, 56]           4,096
      BatchNorm2d-58           [-1, 64, 56, 56]             128
             ReLU-59           [-1, 64, 56, 56]               0
           Conv2d-60           [-1, 64, 56, 56]          36,864
           Conv2d-61           [-1, 64, 56, 56]           4,096
      BatchNorm2d-62           [-1, 64, 56, 56]             128
             ReLU-63           [-1, 64, 56, 56]               0
        Resnet_v2-64           [-1, 64, 56, 56]               0
  ConvTranspose2d-65         [-1, 64, 112, 112]          36,864
           Conv2d-66         [-1, 64, 112, 112]          36,864
           Conv2d-67         [-1, 64, 112, 112]           4,096
      BatchNorm2d-68         [-1, 64, 112, 112]             128
             ReLU-69         [-1, 64, 112, 112]               0
           Conv2d-70         [-1, 64, 112, 112]          36,864
           Conv2d-71         [-1, 64, 112, 112]           4,096
      BatchNorm2d-72         [-1, 64, 112, 112]             128
             ReLU-73         [-1, 64, 112, 112]               0
        Resnet_v2-74         [-1, 64, 112, 112]               0
  ConvTranspose2d-75         [-1, 32, 224, 224]          18,432
           Conv2d-76         [-1, 32, 224, 224]           9,216
           Conv2d-77         [-1, 32, 224, 224]           1,024
      BatchNorm2d-78         [-1, 32, 224, 224]              64
             ReLU-79         [-1, 32, 224, 224]               0
           Conv2d-80         [-1, 32, 224, 224]           9,216
           Conv2d-81         [-1, 32, 224, 224]           1,024
      BatchNorm2d-82         [-1, 32, 224, 224]              64
             ReLU-83         [-1, 32, 224, 224]               0
        Resnet_v2-84         [-1, 32, 224, 224]               0
           Conv2d-85          [-1, 1, 224, 224]              32
  ConvTranspose2d-86          [-1, 256, 14, 14]       1,179,648
           Conv2d-87          [-1, 256, 14, 14]         589,824
           Conv2d-88          [-1, 256, 14, 14]          65,536
      BatchNorm2d-89          [-1, 256, 14, 14]             512
             ReLU-90          [-1, 256, 14, 14]               0
           Conv2d-91          [-1, 256, 14, 14]         589,824
           Conv2d-92          [-1, 256, 14, 14]          65,536
      BatchNorm2d-93          [-1, 256, 14, 14]             512
             ReLU-94          [-1, 256, 14, 14]               0
        Resnet_v2-95          [-1, 256, 14, 14]               0
  ConvTranspose2d-96          [-1, 128, 28, 28]         294,912
           Conv2d-97          [-1, 128, 28, 28]         147,456
           Conv2d-98          [-1, 128, 28, 28]          16,384
      BatchNorm2d-99          [-1, 128, 28, 28]             256
            ReLU-100          [-1, 128, 28, 28]               0
          Conv2d-101          [-1, 128, 28, 28]         147,456
          Conv2d-102          [-1, 128, 28, 28]          16,384
     BatchNorm2d-103          [-1, 128, 28, 28]             256
            ReLU-104          [-1, 128, 28, 28]               0
       Resnet_v2-105          [-1, 128, 28, 28]               0
 ConvTranspose2d-106           [-1, 64, 56, 56]          73,728
          Conv2d-107           [-1, 64, 56, 56]          36,864
          Conv2d-108           [-1, 64, 56, 56]           4,096
     BatchNorm2d-109           [-1, 64, 56, 56]             128
            ReLU-110           [-1, 64, 56, 56]               0
          Conv2d-111           [-1, 64, 56, 56]          36,864
          Conv2d-112           [-1, 64, 56, 56]           4,096
     BatchNorm2d-113           [-1, 64, 56, 56]             128
            ReLU-114           [-1, 64, 56, 56]               0
       Resnet_v2-115           [-1, 64, 56, 56]               0
 ConvTranspose2d-116         [-1, 64, 112, 112]          36,864
          Conv2d-117         [-1, 64, 112, 112]          36,864
          Conv2d-118         [-1, 64, 112, 112]           4,096
     BatchNorm2d-119         [-1, 64, 112, 112]             128
            ReLU-120         [-1, 64, 112, 112]               0
          Conv2d-121         [-1, 64, 112, 112]          36,864
          Conv2d-122         [-1, 64, 112, 112]           4,096
     BatchNorm2d-123         [-1, 64, 112, 112]             128
            ReLU-124         [-1, 64, 112, 112]               0
       Resnet_v2-125         [-1, 64, 112, 112]               0
 ConvTranspose2d-126         [-1, 32, 224, 224]          18,432
          Conv2d-127         [-1, 32, 224, 224]           9,216
          Conv2d-128         [-1, 32, 224, 224]           1,024
     BatchNorm2d-129         [-1, 32, 224, 224]              64
            ReLU-130         [-1, 32, 224, 224]               0
          Conv2d-131         [-1, 32, 224, 224]           9,216
          Conv2d-132         [-1, 32, 224, 224]           1,024
     BatchNorm2d-133         [-1, 32, 224, 224]              64
            ReLU-134         [-1, 32, 224, 224]               0
       Resnet_v2-135         [-1, 32, 224, 224]               0
          Conv2d-136          [-1, 1, 224, 224]              32
================================================================
Total params: 12,384,576
Trainable params: 12,384,576
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.15
Forward/backward pass size (MB): 455.16
Params size (MB): 47.24
Estimated Total Size (MB): 503.56
----------------------------------------------------------------
```

## Loss function:
For mask generation the used loss fuction is Dice Loss. Below is the code for Dice Loss:
```python
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

```
## Complete loss function for mask generation:
```python
from torch.nn import functional as F

def cal_mask_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)


    return loss
```
For more detail about various loss function, click [here](https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/)

## Train the Model:
```python
from tqdm import tqdm
from Losses import *
import torch
from util import save


def train_model(model, criterion, device, train_loader, optimizer, epochs):
  model.train()
  pbar = tqdm(train_loader)
  epoch_samples = 0
  
  for idx, data in enumerate(pbar):
    input_data = torch.cat([data[0], data[1]], dim =1)
    input_data = input_data.to(device)
    mask_label = data[2].to(device)
    depth_label = data[3].to(device)

    # zero the parameter gradients
    optimizer.zero_grad()
    output = model(input_data)

    mask_loss = cal_mask_loss(output[0], mask_label)
    depth_loss = criterion(output[1], depth_label)

    total_loss = loss = 2*mask_loss+ depth_loss
    
    epoch_samples += data[0].size(0)
    epoch_loss = total_loss / epoch_samples

    

    pbar.set_description(desc = f'mask_loss:{mask_loss.item():.4f} depth_loss={depth_loss.item():.4f} Loss = {total_loss.item():.4f} epoch_loss={epoch_loss.item()} ')
    
    total_loss.backward()
    optimizer.step()
  save(output[0].detach().cpu(),f"/content/plots/AfterEpoch{epoch+1}_mask.jpg", 4 )
  save(output[1].detach().cpu(),f"/content/plots/AfterEpoch{epoch+1}_depth.jpg", 4 )
  

  return model, epoch_loss
```

## Model will be trained using the below code:
```python
device = torch.device('cuda')
torch.cuda.current_device()
torch.cuda.get_device_name(0)
print(device)
best_loss=4
lr_scheduler = StepLR(optimizer, step_size=2, gamma = 0.1)
EPOCH=14
for epoch in range(EPOCH):
  
  print('Epoch {}/{}'.format(epoch+1, EPOCH))
  print('-' * 10)
  t0 = time.time()
  model, epoch_loss = train_model(model, criterion, device,train_loader,optimizer,epoch)
  
  if epoch_loss < best_loss:
      print("saving best model")
      best_loss = epoch_loss
      
      torch.save(model.state_dict(), f'/content/drive/My Drive/saved_weight/Weight_after_Epoch_{epoch+1}_{best_loss}.pth')

  time_elapsed = time.time() - t0
  print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  lr_scheduler.step()
  
```
## Core logs generated during the training:

It was **extremely difficult** to train the model with image original size. So, i have **resized the image on 128x128** and trained the Model.
```
Epoch 1/20
----------
mask_loss:0.0391 depth_loss=0.5452 Loss = 0.6235 epoch_loss=2.2266217456490267e-06 : 100%|██████████| 2188/2188 [48:54<00:00,  1.34s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
saving best model
48m 56s
  0%|          | 0/2188 [00:00<?, ?it/s]Epoch 2/20
----------
mask_loss:0.0308 depth_loss=0.5437 Loss = 0.6053 epoch_loss=2.161798420274863e-06 : 100%|██████████| 2188/2188 [45:41<00:00,  1.25s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]saving best model
45m 43s
Epoch 3/20
----------
mask_loss:0.0287 depth_loss=0.5352 Loss = 0.5925 epoch_loss=2.1159087282285327e-06 : 100%|██████████| 2188/2188 [42:45<00:00,  1.17s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]saving best model
42m 47s
Epoch 4/20
----------
mask_loss:0.0274 depth_loss=0.5419 Loss = 0.5967 epoch_loss=2.1310670490493067e-06 : 100%|██████████| 2188/2188 [42:19<00:00,  1.16s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]42m 21s
Epoch 5/20
----------
mask_loss:0.0290 depth_loss=0.5320 Loss = 0.5900 epoch_loss=2.1070134152978426e-06 : 100%|██████████| 2188/2188 [41:25<00:00,  1.14s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]saving best model
41m 26s
Epoch 6/20
----------
mask_loss:0.0291 depth_loss=0.5435 Loss = 0.6017 epoch_loss=2.1488124275492737e-06 : 100%|██████████| 2188/2188 [40:41<00:00,  1.12s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]40m 42s
Epoch 7/20
----------
mask_loss:0.0282 depth_loss=0.5390 Loss = 0.5953 epoch_loss=2.1260884750518017e-06 : 100%|██████████| 2188/2188 [38:48<00:00,  1.06s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 49s
Epoch 8/20
----------
mask_loss:0.0300 depth_loss=0.5400 Loss = 0.6000 epoch_loss=2.1428427317005116e-06 : 100%|██████████| 2188/2188 [38:16<00:00,  1.05s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 18s
Epoch 9/20
----------
mask_loss:0.0300 depth_loss=0.5322 Loss = 0.5921 epoch_loss=2.114779363182606e-06 : 100%|██████████| 2188/2188 [38:39<00:00,  1.06s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 40s
Epoch 10/20
----------
mask_loss:0.0301 depth_loss=0.5412 Loss = 0.6014 epoch_loss=2.147972281818511e-06 : 100%|██████████| 2188/2188 [37:59<00:00,  1.04s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 1s
Epoch 11/20
----------
mask_loss:0.0294 depth_loss=0.5249 Loss = 0.5837 epoch_loss=2.084640073007904e-06 : 100%|██████████| 2188/2188 [37:59<00:00,  1.04s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]saving best model
38m 1s
Epoch 12/20
----------
mask_loss:0.0274 depth_loss=0.5315 Loss = 0.5863 epoch_loss=2.0938193756592227e-06 : 100%|██████████| 2188/2188 [38:38<00:00,  1.06s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 39s
Epoch 13/20
----------
mask_loss:0.0284 depth_loss=0.5392 Loss = 0.5961 epoch_loss=2.128852202076814e-06 : 100%|██████████| 2188/2188 [38:02<00:00,  1.04s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 3s
Epoch 14/20
----------
mask_loss:0.0293 depth_loss=0.5400 Loss = 0.5987 epoch_loss=2.1382231807365315e-06 : 100%|██████████| 2188/2188 [38:49<00:00,  1.06s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 50s
```

## After this I have traind the model again for few epochs by using the last best loss model weight :
```
model.load_state_dict(torch.load('/content/drive/My Drive/saved_weight/Weight_after_Epoch_11_2.084640073007904e-06.pth'))
```

```
Epoch 1/20
----------
mask_loss:0.0391 depth_loss=0.5452 Loss = 0.6235 epoch_loss=2.2266217456490267e-06 : 100%|██████████| 2188/2188 [48:54<00:00,  1.34s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
saving best model
48m 56s
  0%|          | 0/2188 [00:00<?, ?it/s]Epoch 2/20
----------
mask_loss:0.0308 depth_loss=0.5437 Loss = 0.6053 epoch_loss=2.161798420274863e-06 : 100%|██████████| 2188/2188 [45:41<00:00,  1.25s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]saving best model
45m 43s
Epoch 3/20
----------
mask_loss:0.0287 depth_loss=0.5352 Loss = 0.5925 epoch_loss=2.1159087282285327e-06 : 100%|██████████| 2188/2188 [42:45<00:00,  1.17s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]saving best model
42m 47s
Epoch 4/20
----------
mask_loss:0.0274 depth_loss=0.5419 Loss = 0.5967 epoch_loss=2.1310670490493067e-06 : 100%|██████████| 2188/2188 [42:19<00:00,  1.16s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]42m 21s
Epoch 5/20
----------
mask_loss:0.0290 depth_loss=0.5320 Loss = 0.5900 epoch_loss=2.1070134152978426e-06 : 100%|██████████| 2188/2188 [41:25<00:00,  1.14s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]saving best model
41m 26s
Epoch 6/20
----------
mask_loss:0.0291 depth_loss=0.5435 Loss = 0.6017 epoch_loss=2.1488124275492737e-06 : 100%|██████████| 2188/2188 [40:41<00:00,  1.12s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]40m 42s
Epoch 7/20
----------
mask_loss:0.0282 depth_loss=0.5390 Loss = 0.5953 epoch_loss=2.1260884750518017e-06 : 100%|██████████| 2188/2188 [38:48<00:00,  1.06s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 49s
Epoch 8/20
----------
mask_loss:0.0300 depth_loss=0.5400 Loss = 0.6000 epoch_loss=2.1428427317005116e-06 : 100%|██████████| 2188/2188 [38:16<00:00,  1.05s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 18s
Epoch 9/20
----------
mask_loss:0.0300 depth_loss=0.5322 Loss = 0.5921 epoch_loss=2.114779363182606e-06 : 100%|██████████| 2188/2188 [38:39<00:00,  1.06s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 40s
Epoch 10/20
----------
mask_loss:0.0301 depth_loss=0.5412 Loss = 0.6014 epoch_loss=2.147972281818511e-06 : 100%|██████████| 2188/2188 [37:59<00:00,  1.04s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 1s
Epoch 11/20
----------
mask_loss:0.0294 depth_loss=0.5249 Loss = 0.5837 epoch_loss=2.084640073007904e-06 : 100%|██████████| 2188/2188 [37:59<00:00,  1.04s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]saving best model
38m 1s
Epoch 12/20
----------
mask_loss:0.0274 depth_loss=0.5315 Loss = 0.5863 epoch_loss=2.0938193756592227e-06 : 100%|██████████| 2188/2188 [38:38<00:00,  1.06s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 39s
Epoch 13/20
----------
mask_loss:0.0284 depth_loss=0.5392 Loss = 0.5961 epoch_loss=2.128852202076814e-06 : 100%|██████████| 2188/2188 [38:02<00:00,  1.04s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 3s
Epoch 14/20
----------
mask_loss:0.0293 depth_loss=0.5400 Loss = 0.5987 epoch_loss=2.1382231807365315e-06 : 100%|██████████| 2188/2188 [38:49<00:00,  1.06s/it]
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
  0%|          | 0/2188 [00:00<?, ?it/s]38m 50s
```

## Final Output of the Model:
Here is the output given by the trained Model after 30 epochs:
![final_output](https://github.com/ranjanguddu/Machine-Learning/blob/master/EVA4/Session-15/RESULT_fig.jpg)
