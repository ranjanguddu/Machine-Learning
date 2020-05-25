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
