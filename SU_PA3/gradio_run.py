#required imports
import gradio as gr
import torch
import torchaudio
from model import load_model  # Import your model loading function
from utils import pad
import numpy as np
from torch import Tensor

# Specify the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
# model_path = "path/to/fine_tuned_model.pth"
# model = torch.load(model_path)
model = load_model(device)
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the prediction function
def predict(audio_path):
    # Load the audio file
    audio, sample_rate = torchaudio.load(audio_path)
    
    # Preprocess the audio if necessary (e.g., normalization, resampling)
    # For example, if you want to convert the audio to mono and a specific sample rate:
    # audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    # audio = torchaudio.transforms.ConvertAudioChannels(1)(audio)
    audio = np.array(audio[0])
    audio_pad = pad(audio, 64600)
    audio = Tensor(audio_pad)
    
    # Move the audio to the specified device
    audio = audio.to(device)
    
    # Add batch dimension
    audio = audio.unsqueeze(0)
    
    # Get model prediction
    with torch.no_grad():
        output = model(audio)
    
    # Convert output to probabilities if necessary
    if output.shape[1] == 2:
        output = torch.softmax(output, dim=1)[:, 1]  # Take the probability of the positive class
    else:
        output = torch.sigmoid(output)  # Use sigmoid for binary output
    
    # Threshold the prediction
    prediction = (output >= 0.5).item()
    
    # Convert prediction to label
    if prediction:
        return "Fake"
    else:
        return "Real"

# Gradio interface for audio verification using model 1
# Create a Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(source="upload", type="numpy"),  # Use Gradio audio input
    outputs=gr.Text(),  # Use Gradio text output
    title="Audio Deepfake Detection",
    description="Upload an audio file to predict whether it is real or fake."
)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch()

# output = predict("/media/cvlab/EXT_HARD_DRIVE1/Atharva/SpeechAS3/for-2seconds/testing/fake/file2.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav")
# print(output)
