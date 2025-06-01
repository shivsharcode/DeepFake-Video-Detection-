import torch
from model import get_model, get_loss, get_optimizer
from dataloader import get_dataloaders
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get data loaders
train_loader, val_loader, test_loader = get_dataloaders()

# Get model, loss, and optimizer
model = get_model(device)
criterion = get_loss()
optimizer = get_optimizer(model)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_accuracy = correct / total
        print(f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy:.2f}")

    #  NEW
    # Save the model
    torch.save(model.state_dict(), "deepfake_model.pth") # .pth stands for pytorch model file
    print("Model saved successfully.")
    
    
# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)
