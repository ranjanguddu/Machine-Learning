
from util import unnormalize, unnormalize_1D
import matplotlib.pyplot as plt
import torch


def Result_Plot(test_loader, model, device, img_mean, img_std):
  #fig, ax = plt.subplots(nrows=10, ncols=5, figsize=(30,30))
  fig = plt.figure(figsize=(30,30))
  fig.subplots_adjust(hspace=0.1, wspace=0.000001)
  # Get the first batch
  data = next(iter(test_loader))
  input_data = torch.cat([data[0], data[1]], dim =1)
  input_data = input_data.to(device)
  mask_label = data[2].to(device)
  depth_label = data[3].to(device)
  # Predict
  pred = model(input_data)
  ctr=1

  #fig.suptitle("Result os the Trained Model on Test Images", fontweight="bold", fontsize=16)
  for i in range(5):
    plt.axis('off')
    fg_bg = unnormalize(data[1][i],img_mean, img_std)
    plt.subplot(10, 5, ctr)
    plt.imshow(fg_bg)
    if ctr<6 :
      plt.title("FG_BG IMAGE", fontweight="bold", fontsize=20)
    plt.axis('off')
    ctr+=1

    label_mask = unnormalize_1D(data[2][i])
    plt.subplot(10,5, ctr)
    plt.imshow(label_mask, cmap='gray')
    if ctr<6 :
      plt.title("Ground truth: MASK", fontweight="bold", fontsize=20)
    plt.axis('off')
    ctr += 1

    label_depth = unnormalize_1D(data[3][i])
    plt.subplot(10,5, ctr)
    plt.imshow(label_depth, cmap='gray')
    if ctr<6 :
      plt.title("Ground truth: DEPTH", fontweight="bold", fontsize=20)

    plt.axis('off')
    ctr += 1

    pred_mask = unnormalize_1D(pred[0][i])
    plt.subplot(10,5, ctr)
    plt.imshow(pred_mask, cmap='gray')
    if ctr<6 :
      plt.title("Predicted MASK", fontweight="bold", fontsize=20)

    plt.axis('off')
    ctr += 1

    pred_depth = unnormalize_1D(pred[1][i])
    plt.subplot(10,5, ctr)
    plt.imshow(pred_depth, cmap='gray')
    if ctr<6 :
      plt.title("Predicted DEPTH", fontweight="bold", fontsize=20)

    plt.axis('off')
    ctr += 1


  #plt.tight_layout()
  #plt.subplots_adjust(wspace=0, hspace=0)

  plt.savefig('/content/RESULT_fig.jpg')
  plt.show()