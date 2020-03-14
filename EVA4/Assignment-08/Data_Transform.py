import torchvision.transforms as transforms

class Augmentation:
  
  def __init__(self):
    self.transform = transforms.Compose([
      transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
      transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])

  def getTransformedTrain(self):
    return self.transform


