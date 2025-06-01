processed_path_train = 'big-dataset/FaceForensics_compressed/processed_frames/train'
processed_path_val = 'big-dataset/FaceForensics_compressed/processed_frames/val'
processed_path_test = 'big-dataset/FaceForensics_compressed/processed_frames/test'

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #new
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # new end
])

# Load datasets
def get_dataloaders(batch_size=32):
    train_dataset = datasets.ImageFolder(root=processed_path_train, transform=transform)
    val_dataset = datasets.ImageFolder(root=processed_path_val, transform=transform)
    #new
    test_dataset = datasets.ImageFolder(root=processed_path_test, transform=transform)
    #new end
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #new
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #new end
    return train_loader, val_loader, test_loader


