import torch.nn as nn
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Type
from torch import flatten
import numpy as np
import math
import os

np.random.seed(42)
torch.manual_seed(42)

### Downloading the data
            
class DataTransformation:
    
    def __init__(self, size=224):
        
        """Data Transformation to Imagenet images"""
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(p=0.5),
            # Added New transformations
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5, hue=0.2)],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.1),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                std =[0.229,0.224,0.225])
        ])
        
    def __call__(self,x):
        return self.transform(x)
    
def get_val_transform():
    """Deterministic: Resize sides to 256 – center at 224"""
    return transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
        

def get_ImageNet_500(batch_size = 256, num_workers = 4):
    "getting the ImageNet dataset and converting it into a subset based on order"    
    TRAIN_DIR = "/train"
    VAL_DIR = "/val"
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=TRAIN_DIR,
        transform = DataTransformation(size=224)
    )
    
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=VAL_DIR,
        transform=get_val_transform()
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers)

    print(f"train size: {len(train_dataset)}")    
    return train_loader, val_loader



# --- PART B: GLOBAL NORM ---

class GlobalNorm(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x**2, dim=(1,2,3), keepdim=True))
        return x / (norm + 1e-6)

    
class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None
    ):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.gn1 = GlobalNorm()
        
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.gn2 = GlobalNorm()

        
    def forward(self, x: Tensor) -> Tensor: 
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # added GlobalNorm
        out = self.gn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None: 
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        out = self.gn2(out)
        
        return out 
        
    
class ResNet(nn.Module):
    def __init__(self, 
                 img_channels: int,
                 num_layers: int,
                 block: Type[BasicBlock],
                 num_classes: int = 500):
        
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        # Convolution 
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
         
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.gn = GlobalNorm()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer_(block, 64, 2)
        self.layer2 = self.make_layer_(block, 128, 2, stride = 2)
        self.layer3 = self.make_layer_(block, 256, 2, stride = 2)
        self.layer4 = self.make_layer_(block, 512, 2, stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        
    
    def make_layer_(self, block: Type[BasicBlock], out_channels: int , blocks: int, stride: int =1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [block(self.in_channels, out_channels, stride , downsample)]
        
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self,x:Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.gn(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.fc(x)
    
        return x 


# Top-K Metric

def top_k_acc(output: Tensor, target: Tensor, k: int) -> float:
    """computing top k accuracy.
    Top 1 - best guess
    Top 5 - correct label in the first 5 options"""
    
    with torch.no_grad():
        batch_size = target.size(0)
        # idx of top k preds for each sample 
        values, pred = output.topk(k,dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        return (correct_k / batch_size).item() * 100.0
        


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=500).to(device)
    
    # Data
    train_loader, val_loader = get_ImageNet_500(batch_size=256)
    
    # Loss and Optim
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=1e-4)
    
    # Efficiency Technique 1: LR Scheduler - Cosine Annealing
    # T_max = n_epochs - controlling the period of the cosine curve
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)
    
    # Efficiency Technique 2: Mixed Precision Training - Faster on GPU
    scaler = torch.cuda.amp.GradScaler()
    
    # Checkpoints
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Efficiency Technique 3: Early Stopping (ES)
    best_val_acc = 0.0
    patience = 5
    patience_count = 0
    n_epochs = 100
    
    # History
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'val_top5':   []
    }
    
    # Accuracy thresholds to record epoch number (for convergence comparison)
    thresholds     = [10, 20, 30, 40]
    threshold_epochs = {}
    
    ##--------------- Training Loop ----------------------
    
    for epoch in range(n_epochs):
        print(f'\nEpoch [{epoch+1}/{n_epochs}]')
        
        # Training
        model.train()
        train_loss = 0 
        train_correct = 0 
        train_total = 0 
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # gpu
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
            train_loss += loss.item()
            
            # top 1 acc
            _, predicted = output.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'  Batch [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.4f}')
                
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss    = 0.0
        val_top1    = 0.0
        val_top5    = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                loss = criterion(output, labels)

                val_loss += loss.item()
                val_top1 += top_k_acc(output, labels, k=1)
                val_top5 += top_k_acc(output, labels, k=5)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_top1 = val_top1 / len(val_loader)
        avg_val_top5 = val_top5 / len(val_loader)

        # Step scheduler after each epoch
        scheduler.step()
    
        # Metrics History
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_top1)
        history['val_top5'].append(avg_val_top5)
        
        print(f'Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f} | Val Top-1: {avg_val_top1:.2f}% | Val Top-5: {avg_val_top5:.2f}%')
        
        
        # epoch to threshold table
        for t in thresholds:
            if t not in threshold_epochs and avg_val_top1 >= t:
                threshold_epochs[t] = epoch + 1
                print(f"{t}% top 1 acc at epoch {epoch+1}")
        if avg_val_top1 > best_val_acc:
            best_val_acc = avg_val_top1
            torch.save({
                'epoch':      epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_top1':   avg_val_top1,
                'val_top5':   avg_val_top5,
            }, os.path.join(save_dir, 'best_model.pth'))
            patience_count = 0
            print(f'Saved best model - with val acc: {best_val_acc:.2f}%)')
        else:
            patience_count += 1
            
        # Early Stopping
        if patience_count >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
            
    # Final summary
    print('\nTraining Completed')
    print(f'Best Val Top-1: {best_val_acc:.2f}%')
    print(f'Threshold epochs: {threshold_epochs}')


if __name__ == '__main__':
    main()

    
