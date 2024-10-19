from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (appeared to be popular)
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet/resnet stats(appeared to be popular)
])

dataset = datasets.ImageFolder(root='E:\docs\screenshot dataset stuff\data', transform=transform)

# Split dataset into train and test sets and create dataloaders
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # Remaining 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Output set sizes
print(f"Training set size: {len(train_dataset)}")
print(f"Testing set size: {len(test_dataset)}")