import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from cnn_Model import ScreenshotToneClassifier
from data_prep import train_loader, test_loader

# Ensure GPU is used by model instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ScreenshotToneClassifier()
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# TRAINING
num_epochs = 5
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Move inputs and labels to GPU (if available)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: Compute predicted labels by passing inputs to the model
        outputs = model(inputs)

        # Compute the loss between predicted and actual labels
        loss = criterion(outputs, labels)

        # Backward pass: Compute gradients
        loss.backward()

        # Optimize the weights
        optimizer.step()

        # Update running loss for every batch
        running_loss += loss.item()

        # Calculate accuracy on the training set
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

        # Print stats every 100 batches
        if batch_idx % 100 == 32:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)


    model.eval()
    test_running_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss = criterion(outputs, labels)
            test_running_loss += test_loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    epoch_test_loss = test_running_loss / len(test_loader)
    epoch_test_acc = correct_test / total_test
    test_losses.append(epoch_test_loss)
    test_accuracies.append(epoch_test_acc)

    # Save model state that achieves lowest loss
    best_test_loss = float('inf')
    if epoch_test_loss < best_test_loss:
        best_val_loss = epoch_test_loss
        torch.save(model.state_dict(), '../saved models/best_model.pth')

    print(f'Accuracy of the model on test images: {100 * correct_test / total_test:.2f}%')

print('Finished Training')


# Plot training & validation accuracy
epochs_range = range(num_epochs)

plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.show()

plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()