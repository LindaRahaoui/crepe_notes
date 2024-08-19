import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import onsetCNN

def load_model(model_path, device):
    model = onsetCNN().double().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_audio(audio_path, sr=44100, n_fft=1024, hop_length=441, n_mels=80, fmin=27.5, fmax=16000):
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spectrogram1 = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spectrogram2 = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spectrogram3 = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=4096, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)

    mel_spectrogram1_db = librosa.power_to_db(mel_spectrogram1, ref=np.max)
    mel_spectrogram2_db = librosa.power_to_db(mel_spectrogram2, ref=np.max)
    mel_spectrogram3_db = librosa.power_to_db(mel_spectrogram3, ref=np.max)

    return mel_spectrogram1_db, mel_spectrogram2_db, mel_spectrogram3_db

def predict_onsets(model, mel_spectrogram1_db, mel_spectrogram2_db, mel_spectrogram3_db, device, hop_length=441, sr=44100):
    contextlen = 7  # +- frames
    duration = 2 * contextlen + 1
    segment_length = duration  # As used during training
    num_segments = mel_spectrogram1_db.shape[1] - segment_length + 1

    onsets = np.zeros(mel_spectrogram1_db.shape[1])

    with torch.no_grad():
        for i in range(num_segments):
            segment1 = mel_spectrogram1_db[:, i:i+segment_length]
            segment2 = mel_spectrogram2_db[:, i:i+segment_length]
            segment3 = mel_spectrogram3_db[:, i:i+segment_length]
            segment = np.stack([segment1, segment2, segment3], axis=0)
            segment = torch.tensor(segment, dtype=torch.float64).unsqueeze(0).to(device)  # Add batch dimension
            prediction = model(segment).squeeze().cpu().numpy()
            onsets[i:i+segment_length] += prediction
    onsets = onsets * 10

    onsets = np.where(onsets > 0.02, 1, 0)
    return onsets

def plot_onsets(onsets, audio_path, sr=44100, hop_length=441):
    y, sr = librosa.load(audio_path, sr=sr)
    times = np.arange(len(y)) / sr

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(times, y, label='Waveform')
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    onset_times = np.arange(len(onsets)) * hop_length / sr

    plt.subplot(2, 1, 2)
    plt.plot(onset_times, onsets, label='Onset Predictions')

    plt.title('Onset Predictions')
    plt.xlabel('Time (s)')
    plt.ylabel('Onset Confidence')
    plt.legend()

    plt.tight_layout()
    plt.show()

def detect_onsets_linda(audio_path, model_path, save_analysis_files):
    print("Prédictions with my cnn")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    mel_spectrogram1_db, mel_spectrogram2_db, mel_spectrogram3_db = preprocess_audio(audio_path)
    onsets = predict_onsets(model, mel_spectrogram1_db, mel_spectrogram2_db, mel_spectrogram3_db, device)
    
    if save_analysis_files:
        audio_dir = os.path.dirname(audio_path)
        onsets_folder = os.path.join(audio_dir, 'Onsets')

        if not os.path.exists(onsets_folder):
            os.makedirs(onsets_folder)
        
        save_path = os.path.basename(audio_path)[:-4]  # Remove file extension
        model_name = os.path.basename(model_path)[:-3]  # Remove '.pt' or similar extension

        save_path = f"{save_path}_{model_name}_onsets.txt"
        save_path = os.path.join(onsets_folder, save_path)
        np.savetxt(save_path, onsets)
        print(f"Prédictions sauvegardées dans {save_path}")
    
    return onsets
