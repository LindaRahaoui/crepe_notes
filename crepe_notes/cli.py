"""Console script for crepe_notes."""
import pathlib
import click
from pathlib import Path
import pretty_midi as pm
from crepe_notes import process, parse_f0, run_crepe
from tqdm import tqdm 
import warnings
warnings.filterwarnings("ignore")  


import os
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import tensorflow as tf

sys.stderr = stderr
@click.command()

@click.option('--output-label', default='_transcription')
@click.option('--save-dir', default=None, help='Directory to save predictions (default: create a "Prédictions" folder in the audio directory)')
@click.option('--midi_tempo', default=203)
@click.option('--sensitivity', type=click.FloatRange(0, 1), default=0.001)
@click.option('--min-duration', type=click.FloatRange(0, 1), default=0.03, help='Minimum duration of a note in seconds')
@click.option('--min-velocity', type=click.IntRange(0, 127, clamp=True), default=6, help='Minimum velocity of a note in midi scale (0-127)')
@click.option('--disable-splitting', is_flag=True, default=False, help='Disable detection of repeated notes via onset detection')
@click.option('--tuning-offset', type=click.FloatRange(-100, 100, clamp=True), default=False, help='Manually apply a tuning offset in cents. Fractional numbers are allowed. Set to 0 for no offset, otherwise it will be calculated automatically.')
@click.option('--use-smoothing', is_flag=True, default=False, help='Enable smoothing of confidence')
@click.option('--use-cwd', is_flag=True, default=False, help='If True, write to the cwd of the current command, else write to the parent folder of the f0_path')
@click.option('--f0', type=click.Path(exists=True))
@click.option('--save-analysis-files', is_flag=True, default=False, help='Save f0, madmom onsets and amp envelope as files')
@click.option('--not-combined-file', is_flag=True, default=True, help='Save the prediction into one file combined')
@click.option('--post-process', is_flag=True, default=False, help='Save the prediction into one file combined')
@click.option('--save_onsets', is_flag=True, default=False, help='Save the onsets prediction into txt')
@click.option('--my-cnn', is_flag=True, default=False, help='Use author CNN than Madmom')
@click.option('--model-path', default=None, help='Directory of the model CNN for onset detection')
@click.argument('audio_path', type=click.Path(exists=True, path_type=pathlib.Path))
@click.help_option()

def main(f0, audio_path,model_path, output_label, save_dir, not_combined_file, midi_tempo, sensitivity, min_duration, min_velocity, disable_splitting, tuning_offset, use_smoothing, use_cwd, save_analysis_files, post_process, save_onsets,my_cnn):
    if post_process:
      print("POST PROCESS ON")
    # Définir le répertoire de sauvegarde par défaut
    if save_dir is None:
        folder_name = "Prédictions_post_proc" if post_process else "Prédictions"
        save_dir = audio_path.parent / folder_name if audio_path.is_file() else audio_path / folder_name
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if audio_path.is_dir():
        audio_files = list(audio_path.glob('*.wav'))

        # Vérifier si le dossier est vide
        if not audio_files:
            print(f"Erreur : Aucun fichier audio trouvé dans le dossier {audio_path}")
            return  # Arrêter l'exécution du programme

        output_midi = pm.PrettyMIDI(initial_tempo=midi_tempo)
        instrument = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'))

        # Liste des fichiers .wav à traiter
        audio_files = list(audio_path.glob('*.wav'))
        if not_combined_file:
            print("Combined MIDI")
            all_notes = []
            with tqdm(total=len(audio_files), desc="Processing files", unit="file", bar_format='{l_bar}{bar} {percentage:3.0f}%') as pbar:
                
                for audio in audio_files:
                    audio = Path(audio)
                    notes, filtered_amp_envelope = process_audio(audio, f0, model_path,output_label, sensitivity, min_duration, min_velocity, disable_splitting, tuning_offset, use_smoothing, use_cwd, save_analysis_files, save_onsets,my_cnn)
                    
                    if post_process:
                        notes = post_process_notes(notes)
                    
                    all_notes.extend(notes)
                    pbar.update(1)  # Mise à jour de la barre de progression après chaque fichier

            transcribe_audio(all_notes, filtered_amp_envelope, output_midi, instrument, save_dir, "Combined_" + output_label, audio_path, direction=True)
        else:
            print("Not combined MIDI")
            with tqdm(total=len(audio_files), desc="Processing files", unit="file", bar_format='{l_bar}{bar} {percentage:3.0f}%') as pbar:
                for audio in audio_files:
                    output_midi = pm.PrettyMIDI(initial_tempo=midi_tempo)
                    instrument = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'))
                    audio = Path(audio)
                    notes, filtered_amp_envelope = process_audio(audio, f0, model_path, output_label, sensitivity, min_duration, min_velocity, disable_splitting, tuning_offset, use_smoothing, use_cwd, save_analysis_files, save_onsets,my_cnn)
                    
                    if post_process:
                        notes = post_process_notes(notes)
                    
                    transcribe_audio(notes, filtered_amp_envelope, output_midi, instrument, save_dir, output_label, audio)
                    pbar.update(1)  # Mise à jour de la barre de progression après chaque fichier
    else:
        print("single file")
        output_midi = pm.PrettyMIDI(initial_tempo=midi_tempo)
        instrument = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'))
        audio_path = Path(audio_path)
        notes, filtered_amp_envelope = process_audio(audio_path, f0, model_path, output_label, sensitivity, min_duration, min_velocity, disable_splitting, tuning_offset, use_smoothing, use_cwd, save_analysis_files, save_onsets,my_cnn)
        
        if post_process:
            notes = post_process_notes(notes)
        
        transcribe_audio(notes, filtered_amp_envelope, output_midi, instrument, save_dir, output_label, audio_path) 

def process_audio(audio_path, f0, model_path, output_label, sensitivity, min_duration, min_velocity, disable_splitting, tuning_offset, use_smoothing, use_cwd, save_analysis_files, save_onsets,my_cnn):
   
    default_f0_path = audio_path.parent / "F0" / audio_path.with_suffix(".f0.csv").name
    if not default_f0_path.exists() and f0 is None:
        frequency, confidence = run_crepe(audio_path)
    elif f0 is None:
        frequency, confidence = parse_f0(default_f0_path)
    else:
        frequency, confidence = parse_f0(f0)

    notes, filtered_amp_envelope = process(frequency, confidence, audio_path,model_path, sensitivity=sensitivity, use_smoothing=use_smoothing,
            min_duration=min_duration, min_velocity=min_velocity, disable_splitting=disable_splitting, use_cwd=use_cwd,
            tuning_offset=tuning_offset, save_analysis_files=save_analysis_files, save_onsets=save_onsets,my_cnn=my_cnn)
    return notes, filtered_amp_envelope

def transcribe_audio(notes, filtered_amp_envelope, output_midi, instrument, save_dir, output_label,audio_path, direction=False):

    for n in notes:
        if n['start'] >= n['finish']:
            continue

        instrument.notes.append(
            pm.Note(start=n['start'],
                    end=n['finish'],
                    pitch=n['pitch'],
                    velocity=n['velocity']))

    end_audio = len(filtered_amp_envelope) * 0.01

    if len(notes) != 0:
        note = pm.Note(
            velocity=0,
            pitch=0,
            start=notes[-1]['finish'],
            end=end_audio)

        instrument.notes.append(note)
    output_midi.instruments.append(instrument)
    
    if direction :
      output_midi.write(f'{save_dir}/{output_label}.mid')
      print(f"saving midi to {save_dir}/{output_label}.mid")
    else :
      output_midi.write(f'{save_dir}/{audio_path.stem + output_label}.mid')
      print(f'saving midi to :{save_dir}/{audio_path.stem + output_label}.mid')

def post_process_notes(notes, duration_threshold=0.05, velocity_threshold_min=20,velocity_threshold_max=120, pitch_ranges_to_ignore=None):
    """
    Applique un post-traitement aux notes pour supprimer les faux positifs potentiels.

    Args:
        notes (list of dict): Liste des notes détectées.
        duration_threshold (float): Seuil minimum pour la durée des notes à conserver (en secondes).
        velocity_threshold (int): Seuil minimum pour la vélocité des notes à conserver.
        pitch_ranges_to_ignore (list of tuple): Liste de tuples définissant les plages de pitch à ignorer [(min_pitch, max_pitch)].

    Returns:
        list of dict: Liste des notes après post-traitement.
    """
    processed_notes = []

    for note in notes:
        # Calculer la durée de la note
        note_duration = note['finish'] - note['start']
        
        # Appliquer les règles de filtrage
        if note_duration < duration_threshold:
            continue  # Ignorer les notes trop courtes
        
        if note['velocity'] < velocity_threshold_min:
            continue  # Ignorer les notes avec vélocité trop faible
        if note['velocity'] > velocity_threshold_max:
            continue  # Ignorer les notes avec vélocité trop faible
        
        if pitch_ranges_to_ignore:
            ignore_note = False
            for pitch_range in pitch_ranges_to_ignore:
                if pitch_range[0] <= note['pitch'] <= pitch_range[1]:
                    ignore_note = True
                    break
            if ignore_note:
                continue  # Ignorer les notes dont le pitch est dans une plage à ignorer

        # Si la note passe tous les filtres, on la garde
        processed_notes.append(note)
    
    return processed_notes
if __name__ == "__main__":
 
    main()  # pragma: no cover
 
