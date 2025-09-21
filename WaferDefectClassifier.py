import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.multiprocessing as mp
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

#metadata
seed = 69
lr = 1e-3               #learning rate
wd = 1e-5               #weight decay
batch_size = 64
class_no = 36
no_of_epochs = 20

#path
base_path = os.path.dirname(os.path.abspath(__file__))
image_data_path = os.path.join(base_path, "wafer_data", "wafer_tensors.pt") 
label_data_path = os.path.join(base_path, "wafer_data", "label_wm.pt")

#model
class WaferDefectClassifier(nn.Module):
    def __init__(self, classNo = class_no):
        super(WaferDefectClassifier, self).__init__()
        self.conv = nn.Sequential(
            #Layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),    # 56 -> 56
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 56 -> 28
            
            #Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),   # 28 -> 28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 28 -> 14

            #Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 14 -> 14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), #   14 -> 7
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,class_no)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
# transformer class (my name is optimus prime)
# class TransformDataset(Dataset):
#     def __init__(self, tensor, transform = None, class_transform = None):
#         self.tensor = tensor
#         self.transform = transform
#         self.class_transform = class_transform
#         self.normalize = v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

#     def __getitem__(self, index):
#         x, y = self.tensor[0][index], self.tensor[1][index]
#         x = x.float() / 255.0
#         y_index = torch.argmax(y.squeeze(0)).item()

#         if self.class_transform and y_index in self.class_transform:
#             x = self.class_transform[y_index](x)
#         elif self.transform:
#             x = self.transform(x)

#         x = self.normalize(x)
#         return x, y

#     def __len__(self):      #return size of dataset
#         return len(self.tensor[0])

class TransformDataset(Dataset):
    def __init__(self, tensor, transform=None):
        self.tensor = tensor
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.tensor[0][index], self.tensor[1][index]
        x = x.float() / 255.0
        if self.transform:
            x = self.transform(x)
        return x,y
    
    def __len__(self):
        return len(self.tensor[0])

# focal loss class
class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.1, gamma = 0.2, weight = None):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = func.cross_entropy(inputs, targets, weight = self.weight, reduction = "none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

#train per epoch func -
def train_epoch(model, loader, device, criterion, optimizer): #training through one epoch
    model.train()
    total_loss = 0
    correct_predicted = 0
    for image, label in tqdm(loader, desc = "Training"):
        image = image.to(device).float()
        label = torch.argmax(label.to(device).float().squeeze(1), dim=1)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        #loss and accuracy tracking
        total_loss += loss.item()
        _,predicted = torch.max(output,1)
        correct_predicted += (predicted == label).sum().item()

    return total_loss / len(loader), correct_predicted / len(loader.dataset)
        
def eval(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    correct_predicted = 0
    labels = []
    predicteds = []
    with torch.no_grad():
        for image, label in tqdm(loader, desc = "Validation"):
            image = image.to(device).float()
            label = torch.argmax(label.to(device).float().squeeze(1), dim=1)
            output = model(image)
            loss = criterion(output, label)

            #loss and accuracy tracking
            total_loss += loss.item()
            _, predicted = torch.max(output,1)
            correct_predicted += (predicted == label).sum().item()

            labels.extend(label.cpu().numpy())
            predicteds.extend(predicted.cpu().numpy())

    precision = precision_score(labels, predicteds, average='macro', zero_division=0)
    recall = recall_score(labels, predicteds, average='macro', zero_division=0)
    f1 = f1_score(labels, predicteds, average='macro', zero_division=0)
    
    return total_loss / len(loader), correct_predicted / len(loader.dataset), precision, recall, f1, labels, predicteds

# run func
def run():
    
    #check for cuda device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(device)

    #load data
    image_data = torch.load(image_data_path, map_location=torch.device('cpu'))
    label_data = torch.load(label_data_path, map_location=torch.device('cpu'))
    print("Data loaded successfully!")

    #load dataset into python
    index_list = torch.randperm(len(image_data),generator=torch.Generator().manual_seed(seed))

    image_data = image_data[index_list]
    label_data = label_data[index_list]

    #split into test, train, and val - 8:1:1
    tr_size = int(len(image_data) * 0.8)
    test_size = int(len(image_data) * 0.1)
    val_size = len(image_data) - (test_size + tr_size)

    tr_image = image_data[:tr_size]
    tr_label = label_data[:tr_size]
    test_image = image_data[:test_size]
    test_label = label_data[:test_size]
    val_image = image_data[:val_size]
    val_label = label_data[:val_size]
    
    #transforms
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomVerticalFlip(0.5),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomAffine(degrees=12.5, translate=(0.05, 0.05)),
        v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])           
    ])

    # class specific transforms
    # loc_transform = v2.Compose([    #loc
    #     v2.RandomRotation(10),
    #     v2.RandomHorizontalFlip(0.5),
    # ])
    
    # scratch_transform = v2.Compose([
    #     v2.RandomRotation(5),
    #     v2.RandomAffine(degrees=7.5, translate=(0.02, 0.02), shear = 5),
    #     v2.RandomResizedCrop(size=(56,56),scale=[0.9,1]),
    # ])
    
    # edge_transform = v2.Compose([
    #     v2.RandomRotation(12.5),
    #     v2.RandomHorizontalFlip(0.5),
    # ])
    
    # class_transforms = {
    #     0: edge_transform,      #C+EL
    #     1: edge_transform,      #C+EL+L
    #     4: loc_transform,       #C+ER+L
    #     7: loc_transform,       #C+L+EL+S
    #     8: scratch_transform,   #C+L+ER+S
    #     9: scratch_transform,   #C+L+S
    #     12: edge_transform,     #D+EL
    #     14: edge_transform,     #D+EL+S
    #     17: scratch_transform,  #D+ER+S
    #     19: loc_transform,      #D+L+EL+S
    #     20: loc_transform,      #D+L+ER+S
    #     21: loc_transform,      #D+L+S
    #     25: scratch_transform,  #EL+L+S
    #     26: edge_transform,     #EL+S
    #     29: edge_transform,     #Edge-Loc
    #     32: loc_transform,      #Loc
    #     35: scratch_transform,  #Scratch
    # }

    # dataset tranformation
    trans_tr_data = TransformDataset((tr_image, tr_label), transform=transform)  
    trans_test_data = TransformDataset((test_image, test_label), transform=v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
    trans_val_data = TransformDataset((val_image, val_label), transform=v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))

    # load batches
    tr_batch = DataLoader(trans_tr_data, batch_size = batch_size, num_workers = min(4, os.cpu_count() or 1), pin_memory=True, shuffle=True)
    test_batch = DataLoader(trans_test_data, batch_size = batch_size, num_workers = min(4, os.cpu_count() or 1), pin_memory=True, shuffle=False)
    val_batch = DataLoader(trans_val_data, batch_size = batch_size, num_workers = min(4, os.cpu_count() or 1), pin_memory=True, shuffle=False)

    # class weighting
    targets = torch.argmax(tr_label.squeeze(1), dim=1,).numpy()
    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets), dtype=torch.float).to(device)

    # model and optimizer
    model = WaferDefectClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    #criterion = FocalLoss(weight = class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = lrs.LinearLR(optimizer, start_factor=0.05, total_iters=3)
    cosine = lrs.CosineAnnealingLR(optimizer, T_max=no_of_epochs-3)
    scheduler = lrs.SequentialLR(optimizer, schedulers=[warmup,cosine], milestones=[3])

    #training loop
    start_time = time.time()

    #metrics tracking
    l_tr_loss, l_val_loss, l_tr_acc, l_val_acc, l_prec, l_rec, l_f1 = [np.zeros(no_of_epochs) for i in range(7)]

    for epoch in range(no_of_epochs):
        tr_loss, tr_acc = train_epoch(model, tr_batch, device, criterion, optimizer)
        val_loss, val_acc, precision, recall, f1, labels, predicteds = eval(model, val_batch, device, criterion)
        scheduler.step()

        # add metric
        l_tr_loss[epoch] = tr_loss
        l_val_loss[epoch] = val_loss
        l_tr_acc[epoch] = tr_acc
        l_val_acc[epoch] = val_acc
        l_prec[epoch] = precision
        l_rec[epoch] = recall
        l_f1[epoch] = f1

        print(f"\nEpoch no.: {epoch+1}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Train loss: {tr_loss:.6f}")
        print(f"Val loss: {val_loss:.6f}")
        print(f"Train accuracy: {tr_acc * 100:.2f}%")
        print(f"Val accuracy: {val_acc * 100:.2f}%")
        print(f"Precision score: {precision:.4f}")
        print(f"Recall score: {recall:.4f}")
        print(f"F1 score: {f1:.4f}\n")
        print("-" * 40)

    #hasta la vista, baby!
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.3f} seconds")

    #plot graph
    plt.plot(range(1, no_of_epochs + 1), l_tr_loss, label="Train Loss")
    plt.plot(range(1, no_of_epochs + 1), l_val_loss, label="Validation Loss")
    plt.plot(range(1, no_of_epochs + 1), l_tr_acc, label="Train Accuracy")
    plt.plot(range(1, no_of_epochs + 1), l_val_acc, label="Validation Accuracy")
    plt.plot(range(1, no_of_epochs + 1), l_prec, label="Precision")
    plt.plot(range(1, no_of_epochs + 1), l_rec, label="Recall")
    plt.plot(range(1, no_of_epochs + 1), l_f1, label="F1")
    plt.title(f"Training and validation scores for {no_of_epochs} epochs")
    plt.xlabel("Epoch") 
    plt.ylabel("Metric")
    plt.legend()
    plt.show()

    #confusion matrix
    cm = confusion_matrix(labels, predicteds)
    mispreds = []

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i][j] != 0:
                mispreds.append((i, j, cm[i][j]))

    # Sort by decreasing
    mispreds.sort(key=lambda x: x[2], reverse=True)

    print("Most frequent misclassifications (True Class → Predicted Class):")
    for true_cls, pred_cls, count in mispreds:
        print(f"Class {true_cls} -> Class {pred_cls}: {count} times")

    #display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation='vertical')
    plt.title("Validation Set Confusion Matrix")
    plt.show()

#run
if __name__ == "__main__":
    mp.freeze_support()
    run()
