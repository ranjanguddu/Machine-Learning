
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms


class DepthMaskDataset(Dataset):
    def __init__(self, root, image_size, num_data, transform=True):
        
        #csv_file = root/'shuffled_tracker.txt'
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
        
        #self.depth_dir = zipfile.ZipFile(self.root/f'DEPTH_IMAGES/depth_image_batch{index//N+1}.zip')
        #self.imgzip_i = zipfile.ZipFile(self.root/f'FG_BG_IMAGES/image_batch{index//N+1}.zip')
        #self.imgzip_m = zipfile.ZipFile(self.root/f'FG_BG_MASK/mask_batch{index//N+1}.zip')
        #print(f'Value of index chosen:{index} and corresponding folder is: {index//N+1}')
        self.image_folder = self.root/f'image_batch{index//N+1}'
        self.mask_folder = self.root/f'mask_batch{index//N+1}'
        self.depth_folder = self.root/f'depth_image_batch{index//N+1}'
        self.bg_folder = self.root/'street_bg'

        image_path = os.path.join(self.image_folder, self.annotations.iloc[index, 0])
        #print(f'index chosen for image is:{index} coreesponding file:{self.annotations.iloc[index, 0]}')
        #print(f'image_path:{image_path}')
        image = Image.open(image_path)
        
        bg_path = os.path.join(self.bg_folder, self.annotations.iloc[index, 1])
        bg = Image.open(bg_path)
        
        mask_path = os.path.join(self.mask_folder, self.annotations.iloc[index, 2])
        mask = Image.open(mask_path)
        
        depth_path = os.path.join(self.depth_folder, self.annotations.iloc[index, 3])
        depth = Image.open(depth_path)

        '''
        image_path = self.imgzip_i.open(f'image_batch{index//N+1}/{self.annotations.iloc[index, 0]}')
        image = Image.open(image_path)
        mask_path = self.imgzip_m.open(f'mask_batch{index//N+1}/{self.annotations.iloc[index, 2]}')
        mask = Image.open(mask_path)
        depth_path = self.imgzip_d.open(f'depth_image_batch{index//N+1}/{self.annotations.iloc[index, 3]}')
        depth = Image.open(depth_path)
        bg_path = os.path.join(self.bg_files, self.annotations.iloc[index, 1])
        bg = Image.open(bg_path)

        '''

        if self.transform:

            bg_mean = [0.4066298566301917, 0.4002413496649073, 0.39249680184578123]
            bg_std = [0.2516888771733121, 0.25270893840650427, 0.2627707634116584]

            img_mean = [0.406589950038055, 0.3974733626335778, 0.3922238309037007]
            img_std = [0.2537575580108365, 0.2544099256403426, 0.26358312983854404]


            size = self.imsize

            if size == None:

                bg_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(bg_mean, bg_std)])
                img_transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(img_mean, img_std)])
                mask_transform = transforms.Compose([ transforms.ToTensor(), ])
                depth_transform = transforms.Compose([transforms.ToTensor(), ])

                
            else:
                bg_transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(bg_mean, bg_std)])
                img_transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(img_mean, img_std)])
                mask_transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), ])
                depth_transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), ])

            
            bg = bg_transform(bg)
            image= img_transform(image)
            mask = mask_transform(mask)
            depth = depth_transform(depth)
        return(bg, image, mask, depth)