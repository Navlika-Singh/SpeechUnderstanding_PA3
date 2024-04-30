#required imports
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model import load_model
from utils import FORDataset
from main import evaluate_model

# Specify the device (CPU or GPU) for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = load_model(device)

# Move the model to the specified device
model.to(device)

# Define the root directory of the FOR dataset
root_dir = "/media/cvlab/EXT_HARD_DRIVE1/Atharva/SpeechAS3/for-2seconds"

# Define the batch size
batch_size = 32

# Define the number of epochs
num_epochs = 1

# Load the FOR dataset
training_dataset = FORDataset(root_dir, "training")
validation_dataset = FORDataset(root_dir, "validation")
testing_dataset = FORDataset(root_dir, "testing")

# Create DataLoaders
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

# Set up the loss function and optimizer
loss_function = nn.CrossEntropyLoss()  # Use binary cross-entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Fine-tune the model
print("[Training started.]")
for epoch in range(num_epochs):
    print("[EPOCH]: ", epoch, "...")
    model.train()
    epoch_loss = 0
    
    for audio_batch, label_batch in training_loader:
        # Move the data and labels to the specified device
        audio_batch = audio_batch.to(device)
        label_batch = label_batch.to(device)
        
        # Forward pass: compute the model's predictions
        outputs = model(audio_batch)
        
        # Compute the loss
        loss = loss_function(outputs, label_batch)
        
        # Backward pass: compute the gradients
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Compute the gradients
        
        # Update the model's parameters
        optimizer.step()
        
        # Accumulate loss for the epoch
        epoch_loss += loss.item()
    
    # Calculate average loss
    avg_loss = epoch_loss / len(training_loader)
    
    # Validation loop
    model.eval()  # Set model to evaluation mode
    validation_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    # Iterate through the validation data
    with torch.no_grad():
        for audio_batch, label_batch in validation_loader:
            # Move the data and labels to the specified device
            audio_batch = audio_batch.to(device)
            label_batch = label_batch.to(device)
            
            # Forward pass: compute the model's predictions
            outputs = model(audio_batch)
            
            # Compute the loss
            loss = loss_function(outputs, label_batch)
            
            # Accumulate validation loss
            validation_loss += loss.item()
            
            # Convert outputs to binary predictions (threshold at 0.5)
            # Convert model outputs to a 1D tensor if necessary
            if outputs.shape[1] == 2:
                # If the output shape is (batch_size, 2), take the second column (positive class probabilities)
                outputs = outputs[:, 1]
            
            # Convert outputs to binary predictions (threshold at 0.5)
            predictions = (outputs >= 0.5).float()
            
            # Calculate the number of correct predictions
            correct_predictions += (predictions == label_batch).sum().item()
            total_samples += len(label_batch)
    
    # Calculate average validation loss
    avg_validation_loss = validation_loss / len(validation_loader)
    
    # Calculate validation accuracy
    validation_accuracy = correct_predictions / total_samples
    
    # Print average loss for the epoch and validation metrics
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, "
          f"Validation Loss: {avg_validation_loss:.4f}, "
          f"Validation Accuracy: {validation_accuracy:.4f}")
    
# Optional: Save the fine-tuned model
# torch.save(model, "/media/cvlab/EXT_HARD_DRIVE1/Atharva/SU_PA3/models/fine_tuned_model.pth")
print("[Training complete.]")

# Testing loop
model.eval()  # Set model to evaluation mode
test_loss = 0
correct_predictions = 0
total_samples = 0

# Iterate through the test data
print("[Testing]")
with torch.no_grad():
    for audio_batch, label_batch in testing_loader:
        # Move the data and labels to the specified device
        audio_batch = audio_batch.to(device)
        label_batch = label_batch.to(device)
        
        # Forward pass: compute the model's predictions
        outputs = model(audio_batch)
        
        # Compute the loss
        loss = loss_function(outputs, label_batch)
        
        # Accumulate test loss
        test_loss += loss.item()
        
        # Convert outputs to binary predictions (threshold at 0.5)
        # Convert model outputs to a 1D tensor if necessary
        if outputs.shape[1] == 2:
            # If the output shape is (batch_size, 2), take the second column (positive class probabilities)
            outputs = outputs[:, 1]
        
        # Convert outputs to binary predictions (threshold at 0.5)
        predictions = (outputs >= 0.5).float()
        
        # Calculate the number of correct predictions
        correct_predictions += (predictions == label_batch).sum().item()
        total_samples += len(label_batch)
        
# Calculate average test loss
avg_test_loss = test_loss / len(testing_loader)
    
# Calculate test accuracy
test_accuracy = correct_predictions / total_samples
    
# Print average test loss and accuracy
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Evaluate the model
auc, eer = evaluate_model(model, testing_loader)

# Print the AUC and EER
print(f"AUC: {auc}")
print(f"EER: {eer}")
