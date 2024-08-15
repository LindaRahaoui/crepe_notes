import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.signal import find_peaks
import csv
from pathlib import Path
import pretty_midi as pm
import os


def Create_Note_list():
    frequency_note = {
        8.18: 'C-1', 8.661: 'C#-1',8.66:'Db-1', 9.18: 'D-1', 9.721: 'D#-1', 9.72:'Eb-1', 10.3: 'E-1', 10.91: 'F-1', 11.561: 'F#-1', 11.56:'Gb-1',
        12.25: 'G-1', 12.981: 'G#-1', 12.98: 'Ab-1', 13.75: 'A-1', 14.571: 'A#-1', 14.57: 'Bb-1', 15.43: 'B-1', 16.35: 'C0', 17.321: 'C#0', 17.32:'Db0',
        18.35: 'D0', 19.451: 'D#0', 19.45: 'Eb0', 20.6: 'E0', 21.83: 'F0', 23.121: 'F#0', 23.12:'Gb0', 24.5: 'G0', 25.961: 'G#0', 25.96:'Ab0',
        27.5: 'A0', 29.141: 'A#0', 29.14: 'Bb0', 30.87: 'B0', 32.7: 'C1', 34.651: 'C#1', 34.65: 'Db1', 36.71: 'D1', 38.891: 'D#1', 38.89:'Eb1',
        41.2: 'E1', 43.65: 'F1', 46.251: 'F#1', 46.25:'Gb1', 49.0: 'G1', 51.911: 'G#1', 51.91:'Ab1', 55.0: 'A1', 58.271: 'A#1', 58.27: 'Bb1',
        61.74: 'B1', 65.41: 'C2', 69.31: 'C#2', 69.3: 'Db2', 73.42: 'D2', 77.781: 'D#2', 77.78: 'Eb2', 82.41: 'E2', 87.31: 'F2',
        92.51: 'F#2',92.5: 'Gb2', 98.0: 'G2', 103.831: 'G#2', 103.83: 'Ab2', 110.0: 'A2', 116.541: 'A#2', 116.54: 'Bb2', 123.47: 'B2', 130.81: 'C3',
        138.591: 'C#3',138.59:'Db3', 146.83: 'D3', 155.561: 'D#3', 155.56:'Eb3', 164.81: 'E3', 174.61: 'F3', 185.01: 'F#3', 185.0: 'Gb3', 196.0: 'G3',
        207.651: 'G#3',207.65:'Ab3', 220.0: 'A3', 233.081: 'A#3', 233.08:'Bb3', 246.94: 'B3', 261.63: 'C4', 277.181: 'C#4', 277.18:'Db4', 293.66: 'D4',
        311.131: 'D#4',311.13: 'Eb4', 329.63: 'E4', 349.23: 'F4', 369.991: 'F#4', 369.99:'Gb4', 392.0: 'G4', 415.31: 'G#4', 415.3: 'Ab4', 440.0: 'A4',
        466.161: 'A#4',466.16:'Bb4', 493.88: 'B4', 523.25: 'C5', 554.371: 'C#5', 554.37:'Db5', 587.33: 'D5', 622.251: 'D#5', 622.25:'Eb5', 659.25: 'E5',
        698.46: 'F5', 739.991: 'F#5', 739.99:'Gb5', 783.99: 'G5', 830.611: 'G#5', 830.61:'Ab5', 880.0: 'A5', 932.331: 'A#5', 932.33: 'Bb5', 987.77: 'B5',
        1046.5: 'C6', 1108.731: 'C#6', 1108.73: 'Db6', 1174.66: 'D6', 1244.511: 'D#6', 1244.51:'Eb6', 1318.51: 'E6', 1396.91: 'F6', 1479.981: 'F#6', 1479.98: 'Gb6',
        1567.98: 'G6', 1661.221: 'G#6', 1661.22: 'Ab6', 1760.0: 'A6', 1864.661: 'A#6', 1864.66:'Bb6', 1975.53: 'B6', 2093.0: 'C7'
    }
    liste_notes = ['C-1', 'C#-1', 'Db-1', 'D-1', 'D#-1', 'Eb-1', 'E-1', 'F-1', 'F#-1', 'Gb-1', 'G-1', 'G#-1', 'Ab-1', 'A-1', 'A#-1', 'Bb-1', 'B-1', 'C0', 'C#0', 'Db0', 'D0', 'D#0', 'Eb0', 'E0', 'F0', 'F#0', 'Gb0', 'G0', 'G#0', 'Ab0', 'A0', 'A#0', 'Bb0', 'B0', 'C1', 'C#1', 'Db1', 'D1', 'D#1', 'Eb1', 'E1', 'F1', 'F#1', 'Gb1', 'G1', 'G#1', 'Ab1', 'A1', 'A#1', 'Bb1', 'B1', 'C2', 'C#2', 'Db2', 'D2', 'D#2', 'Eb2', 'E2', 'F2', 'F#2', 'Gb2', 'G2', 'G#2', 'Ab2', 'A2', 'A#2', 'Bb2', 'B2', 'C3', 'C#3', 'Db3', 'D3', 'D#3', 'Eb3', 'E3', 'F3', 'F#3', 'Gb3', 'G3', 'G#3', 'Ab3', 'A3', 'A#3', 'Bb3', 'B3', 'C4', 'C#4', 'Db4', 'D4', 'D#4', 'Eb4', 'E4', 'F4', 'F#4', 'Gb4', 'G4', 'G#4', 'Ab4', 'A4', 'A#4', 'Bb4', 'B4', 'C5', 'C#5', 'Db5', 'D5', 'D#5', 'Eb5', 'E5', 'F5', 'F#5', 'Gb5', 'G5', 'G#5', 'Ab5', 'A5', 'A#5', 'Bb5', 'B5', 'C6', 'C#6', 'Db6', 'D6', 'D#6', 'Eb6', 'E6', 'F6', 'F#6', 'Gb6', 'G6', 'G#6', 'Ab6', 'A6', 'A#6', 'Bb6', 'B6', 'C7']
    return liste_notes ,  frequency_note

def get_note_guessed_from_fname(note_list: list, fname: str):
    """Extract MIDI note name based on wav filename. 
    Note name should be in the filename

    Args:
        note_list (list[str]): List of possible notes as strings
        fname (str): name of wav file

    Returns:
        A tuple containing :
            str or None : the note name
            int or None : the note number in pretty_midi format or None if note name was not in file name
    """
    midi_note = None
    note_name = None
    for note_candidate in note_list:
        if note_candidate in fname:
            note_name = note_candidate
            midi_note = pm.note_name_to_number(note_candidate)
            break
    return note_name, midi_note



def detect_onsets(audio_path,save_analysis_files= True, Display=False):
    onsets_path = str(audio_path.with_suffix('.onsets.npz'))
    if not os.path.exists(onsets_path):
        print(f"Onsets file not found at {onsets_path}")
        print("Running onset detection...")
        
        from madmom.features import CNNOnsetProcessor
        
        onset_activations = CNNOnsetProcessor()(str(audio_path))
        if save_analysis_files:
            np.savez(onsets_path, activations=onset_activations)
    else:
        print(f"Loading onsets from {onsets_path}")
        onset_activations = np.load(onsets_path, allow_pickle=True)['activations']

    onsets = np.zeros_like(onset_activations)
    onsets[find_peaks(onset_activations, distance=4, height=0.6)[0]] = 1
    if Display:
        # Afficher les activations des onsets
        plt.figure(figsize=(10, 4))
        plt.plot(onsets)
        plt.title("Activations des onsets")
        plt.xlabel("Échantillons")
        plt.ylabel("Activation")
        plt.show()

    return onsets

def DisplayAudio(audio_path):

    #Afficher le signal audio
    y,fs = librosa.load(audio_path)
    # Définir l'axe temporel en secondes
    t = np.linspace(0, len(y)/fs, len(y))
    plt.figure()
    plt.plot(t,y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()


def save_notes_to_csv(onsets, offsets, pitchs, midi_notes,velocities, file_path):
    """
    Sauvegarde les annotations dans un fichier CSV.

    Parameters:
    onsets (list of float): Liste des temps d'onsets des notes.
    offsets (list of float): Liste des temps d'offsets des notes.
    pitch (float): Pitch de la note en Hz.
    midi_note (int): Note MIDI.
    file_path (str): Chemin du fichier CSV de sortie.
    """
    # Vérifier que les onsets et offsets ont la même longueur
    assert len(onsets) == len(offsets) == len(pitchs)== len(midi_notes), "Les listes d'onsets et d'offsets doivent avoir la même longueur."
    
    # Définir l'en-tête du CSV
    header = ['onset', 'offset', 'pitch', 'midi_note','velocity']
    
    # Créer le dossier si n'existe pas
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Écrire les données dans le fichier CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for onset, offset,pitch,midi_note,velocity in zip(onsets, offsets, pitchs, midi_notes,velocities):
            writer.writerow([onset, offset, pitch, midi_note,velocity])
    print(f"Annotations saved to {file_path}")




def read_onsets(file_path):
    # Lire les onsets depuis un fichier texte
    onsets = np.loadtxt(file_path)
    
    # Transformer les onsets en un array en ligne
    onsets_array = np.array(onsets).reshape(1, -1)
    
    return onsets_array




