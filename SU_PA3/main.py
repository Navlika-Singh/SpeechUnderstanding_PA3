#required imports
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np

from utils import AudioDataset
from model import load_model

#calculate eer
def calculate_eer(labels, scores):
    scores = np.array(scores)
    labels = np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

# Function to evaluate the model
def evaluate_model(model, data_loader):
    # Set the model to evaluation mode
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Iterate through the data loader
    with torch.no_grad():
        for audio_batch, label_batch in data_loader:
            # Move the audio_batch to the device (GPU if available)
            audio_batch = audio_batch.to(device)
            
            # Get model predictions
            outputs = model(audio_batch)
            
            # Apply softmax or any activation function to get probabilities (if needed)
            # Example for binary classification:
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            # Store predictions and labels
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(label_batch.numpy())
    
    # Calculate AUC
    auc = roc_auc_score(all_labels, all_preds)
    
    # Calculate EER
    eer = calculate_eer(all_labels, all_preds)
    
    return auc, eer

#specify device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Specify the root directory of the custom dataset
root_dir = "/media/cvlab/EXT_HARD_DRIVE1/Atharva/SpeechAS3/Dataset_Speech_Assignment"

# Load the pre-trained model
model = load_model(device)

# Load the custom dataset
custom_dataset = AudioDataset(root_dir)

# Create a DataLoader for the custom dataset
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=False)

# Evaluate the model
print("[Evaluating model...]")
auc, eer = evaluate_model(model, data_loader)

# Print the AUC and EER
print(f"AUC: {auc}")
print(f"EER: {eer}")
