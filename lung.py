import monai.networks
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
import torch.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, random_split
import os
import torch
import torchxrayvision
import monai
from monai.networks.nets import ViT
from torch.utils.data import ConcatDataset
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

class dataset(Dataset):

    def __init__(self, x_data, y_data, transform = None):
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        img = self.x_data[idx]
        label = self.y_data[idx]

        if self.transform:
            img = self.transform(img)

        if isinstance(img, np.ndarray):

            img = torch.from_numpy(img).permute(2,0,1)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()
        
        

        return img, label
    
def train(model_list, train_loader, test_loader, val_loader, criterion, num_epochs, device):

    total_step = len(train_loader)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        for model, optimizer, name, adapter, scheduler in model_list:
            
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                images = images.float()

                optimizer.zero_grad()

                with autocast():
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    if adapter is not None:
                        outputs = adapter(outputs)
                                       

                    loss = criterion(outputs, labels)

                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                del images, labels, outputs
            
            if scheduler is not None:
                scheduler.step()
            print('epoch [{}/{}], loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

            
        #individual eval
        for model, optimizer, name, adapter, scheduler in model_list: 
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    images = images.float()
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    if adapter is not None:
                        
                        outputs = adapter(outputs)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print('accuracy: {}%'.format(100 * correct / total))

        #ensemble eval
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                images = images.float()

                ensemble_outputs = None

                total_probs = []

                for model, optimizer, name, adapter, scheduler in model_list:
                    model.eval()
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    if adapter is not None:
                        outputs = adapter(outputs)

                    probs = torch.nn.functional.softmax(outputs, dim = 1)
                    total_probs.append(probs)
                


                    

                ensemble_probs = torch.zeros_like(total_probs[0])
                for probs in total_probs:
                    ensemble_probs += probs   
                ensemble_probs /= len(total_probs)

                _, predicted = torch.max(ensemble_probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'ensemble validation accuracy: {100 * correct / total:.2f}%')

        #eval on test 
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                images = images.float()

                ensemble_outputs = None

                outputs = model(images)

                total_probs = []

                for model, optimizer, name, adapter, scheduler in model_list:
                    model.eval()
                    outputs = model(images)

                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    if adapter is not None:
                        outputs = adapter(outputs)

                    probs = torch.nn.functional.softmax(outputs, dim = 1)
                    total_probs.append(probs)
            
                ensemble_probs = torch.zeros_like(total_probs[0])
                for probs in total_probs:
                    ensemble_probs += probs
                ensemble_probs /= len(total_probs)

                _, predicted = torch.max(ensemble_probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print('accuracy on test images: {}%'.format( 100 * correct / total))





def main():


    num_classes = 3
    num_epochs = 64
    batch_size = 16
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_list = []

    densenet = torchxrayvision.models.DenseNet(weights = "densenet121-res224-all")
    densenet = densenet.to(device)
    densenet_adapter = nn.Linear(18,3).to(device)

    densenet_optim = torch.optim.Adam(params = list(densenet.parameters()) + list(densenet_adapter.parameters()), lr = 0.001)
    



    model_list.append((densenet, densenet_optim, 'densenet', densenet_adapter, None))

    
    vit = ViT(in_channels =1, img_size = (224,224), patch_size = (16,16), 
              hidden_size = 256, mlp_dim = 1024, num_layers = 4, 
              num_heads = 4, classification = True, 
              num_classes = 3, spatial_dims = 2)
    vit = vit.float().to(device) 


    vit_optim = torch.optim.AdamW(vit.parameters(), lr = 1e-5, weight_decay = 0.0001)

    warmup_scheduler_vit = LinearLR(
        vit_optim,
        start_factor = 0.1,
        end_factor = 1.0,
        total_iters = 5
    )

    cosine_scheduler_vit = CosineAnnealingLR(
        vit_optim,
        T_max = num_epochs - 5,
        eta_min = 1e-6
    )

    vit_scheduler = SequentialLR(
        vit_optim,
        schedulers = [warmup_scheduler_vit, cosine_scheduler_vit],
        milestones = [5]
    )

    model_list.append((vit, vit_optim, 'vit', None, vit_scheduler))


    data_dir = "/root/lung cancer dataset/Augmented IQ-OTHNCCD lung cancer dataset"
    categories = ['Benign cases', 'Malignant cases', 'Normal cases']
    
    #train_dataset = []
    #val_dataset = []
    #test_dataset = []

   


    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5],
                             std = [0.5])
    ])

    train_transform = transforms.Compose([

            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees = 15, shear = 10, scale = (0.8,1.2)),
            transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p = 0.3, scale = (0.02, 0.1)),
            transforms.Normalize(mean = [0.5],
                                 std = [0.5])


    ])





    all_images = []
    all_labels = []
    
    for label, category in enumerate(categories):
        path = os.path.join(data_dir, category)

        image_files = [f for f in os.listdir(path) if f.endswith('.jpg')]

        for image_file in image_files:            
            image_path = os.path.join(path, image_file)
            image = Image.open(image_path).convert('L')
            
            image = image.resize((224, 224))
            image_array = np.array(image)
            all_images.append(image_array)
            all_labels.append(label)
        
        

        #train_size = int(0.7 * len(data))
        #val_size = int(0.15 * len(data))
        #test_size = len(data) - train_size - val_size
        #generator = torch.Generator().manual_seed(42)
        #train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size], generator)
        #train_dataset.append(train_data)
        #val_dataset.append(val_data)
        #test_dataset.append(test_data)

    temp_images, test_images, temp_labels, test_labels = train_test_split(
        all_images, all_labels, test_size= 0.3, random_state = 42, stratify = all_labels 
    )

    train_images, val_images, train_labels, val_labels = train_test_split(
        temp_images, temp_labels, test_size = 0.1765, random_state = 42, stratify = temp_labels
    )

    #data = dataset(x_data = all_images, y_data = all_labels, transform = transform)


    train_dataset = dataset(train_images, train_labels, transform)
    val_dataset = dataset(val_images, val_labels, transform)
    test_dataset = dataset(test_images, test_labels, transform)


    #train_size = int(0.7 * len(all_images))
    #val_size = int(0.15 * len(all_images))
    #test_size = len(all_images) - train_size - val_size
    
    #train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size])





    #final_train = ConcatDataset(train_dataset)
    #final_val = ConcatDataset(val_dataset)
    #final_test = ConcatDataset(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    criterion = nn.CrossEntropyLoss()

    train(model_list, train_loader, test_loader, val_loader, criterion, num_epochs, device)



if __name__ == "__main__":
    main()