"""Main module."""
from librosa import load, pitch_tuning, hz_to_midi, time_to_samples
from scipy.signal import find_peaks, hilbert, peak_widths, butter, filtfilt
import numpy as np
import crepe
from scipy.io import wavfile
from pathlib import Path
from fonctions import *
from New_cnn import *
import warnings
warnings.filterwarnings("ignore")  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def detect_onsets(audio_path,save_analysis_files, Display=False):
  
    audio_dir = audio_path.parent
    
    onsets_path = os.path.join(audio_dir, audio_path.stem + '.onsets.npz')
    
    if not os.path.exists(onsets_path):
        print(f"Fichier des onsets non trouvé à {onsets_path}")
        print("Lancement de la détection des onsets...")
        
        from madmom.features import CNNOnsetProcessor
        
        onset_activations = CNNOnsetProcessor()(str(audio_path))
        if save_analysis_files:
            np.savez(onsets_path, activations=onset_activations)
            print(f"Onsets sauvegardés dans {onsets_path}")
    else:
        print(f"Chargement des onsets depuis {onsets_path}")
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


def run_crepe(audio_path):
    
    sr, audio = wavfile.read(str(audio_path))
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
    
    return frequency, confidence


def steps_to_samples(step_val, sr, step_size=0.01):
    return int(step_val * (sr * step_size))


def samples_to_steps(sample_val, sr, step_size=0.01):
    return int(sample_val / (sr * step_size))


def freqs_to_midi(freqs, tuning_offset=0):
    return np.nan_to_num(hz_to_midi(freqs) - tuning_offset, neginf=0)


def calculate_tuning_offset(freqs):
    tuning_offset = pitch_tuning(freqs)
    # print(f"Tuning offset: {tuning_offset * 100} cents")
    return tuning_offset


def parse_f0(f0_path):
    data = np.genfromtxt(f0_path, delimiter=',', names=True)
    return np.nan_to_num(data['frequency']), np.nan_to_num(data['confidence'])
    
def save_f0(f0_path, frequency, confidence):
    np.savetxt(f0_path, np.stack([np.linspace(0, 0.01 * len(frequency), len(frequency)).astype('float'), frequency.astype('float'), confidence.astype('float')], axis=1), fmt='%10.7f', delimiter=',', header='time,frequency,confidence', comments='')
    return

def load_audio(audio_path, cached_amp_envelope_path, default_sample_rate, detect_amplitude, save_amp_envelope):
    if cached_amp_envelope_path.exists():
        # if we have a cached amplitude envelope, no need to load audio
        filtered_amp_envelope = np.load(cached_amp_envelope_path, allow_pickle=True)['filtered_amp_envelope']
        # sr = get_samplerate(audio_path)
        sr = default_sample_rate # this is mainly to make tests work without having to load audio
        y = None
    else:
        try:
            y, sr = load(str(audio_path), sr=None)
        except:
            print("Error loading audio file. Amplitudes will be set to 80")
            detect_amplitude = False
            y = None
            pass

        amp_envelope = np.abs(hilbert(y))

        scaled_amp_envelope = np.interp(amp_envelope, (amp_envelope.min(), amp_envelope.max()), (0, 1))
        # low pass filter the amplitude envelope
        b, a = butter(4, 50, 'low', fs=sr)
        filtered_amp_envelope = filtfilt(b, a, scaled_amp_envelope)[::(sr//100)]
    
    if save_amp_envelope:
        np.savez(cached_amp_envelope_path, filtered_amp_envelope=filtered_amp_envelope)
    
    return sr, y, filtered_amp_envelope, detect_amplitude    



def process(freqs,
            conf,
            audio_path,
            model_path,
            sensitivity=0.001,
            use_smoothing=False,
            min_duration=0.05,
            min_velocity=9,
            disable_splitting=False,
            use_cwd=True,
            tuning_offset=False,
            detect_amplitude=True,
            save_amp_envelope=False,
            default_sample_rate=44100,
            save_analysis_files=False,
            my_cnn=False):
    
    display = False
    # Etape 1 : Chargement de l'audio
    fname = audio_path.stem
    note_list,_ = Create_Note_list()
    cached_amp_envelope_path = audio_path.with_suffix(".amp_envelope.npz")
    sr, y, filtered_amp_envelope, detect_amplitude = load_audio(audio_path, cached_amp_envelope_path, default_sample_rate, detect_amplitude, (save_analysis_files or save_amp_envelope))
    note, midi_note = get_note_guessed_from_fname(note_list=note_list, fname=fname)
    # print(f"Note guessed from filename: {note} ({midi_note})")
    
    if display:
      DisplayAudio(audio_path)

    if use_cwd:
        # write to location that the bin was run from
        output_filename = audio_path.stem
    else:
        # write to same folder as the orignal audio file
        output_filename = str(audio_path.parent) + "/" + audio_path.stem

    
    # Étape 2 : Sauvegarde des fichiers d'analyses (optionnel)
    if save_analysis_files:
        f0_folder_path = audio_path.parent / "F0"
        f0_folder_path.mkdir(parents=True, exist_ok=True)

        # Définir le chemin complet pour le fichier .f0.csv à l'intérieur du dossier "F0"
        f0_path = f0_folder_path / audio_path.with_suffix(".f0.csv").name

        # Vérifier si le fichier .f0.csv n'existe pas déjà et l'enregistrer
        if not f0_path.exists():
            # print(f"Saving f0 to {f0_path}")
            save_f0(f0_path, freqs, conf)

    # Étape 3 : Détection des onsets avec madmom
    if not disable_splitting:
        
        if my_cnn: 
          
            onsets = detect_onsets_linda(audio_path,model_path,save_analysis_files)
        # # Chargemnt des onsets 
        
        else :
          
          onsets = detect_onsets(audio_path, save_analysis_files,Display=False)

    # Étape 4 : Calcul du décalage de l'accordage
    if tuning_offset == False:
        tuning_offset = calculate_tuning_offset(freqs)
    else:
        tuning_offset = tuning_offset / 100
    
    # Étape 5 : Conversion des fréquences en pitch MIDI
    # get pitch gradient
    midi_pitch = freqs_to_midi(freqs)

    
    pitch_changes = np.abs(np.gradient(midi_pitch))# Calcul des changements de pitch
    pitch_changes = np.interp(pitch_changes,
                              (pitch_changes.min(), pitch_changes.max()),
                              (0, 1))# Normalisation des changements de pitch
  
    
    # Étape 6 : Détection des pics de confiance
    # get confidence peaks with peak widths (prominences)
    conf_peaks, conf_peak_properties = find_peaks(1 - conf,
                                                  distance=4,
                                                  prominence=sensitivity)
    
   
    # combine pitch changes and confidence peaks to get change point signal
    change_point_signal = (1 - conf) * pitch_changes
    change_point_signal = np.interp(
        change_point_signal,
        (change_point_signal.min(), change_point_signal.max()), (0, 1))
    peaks, peak_properties = find_peaks(change_point_signal,
                                        distance=4,
                                        prominence=sensitivity)
    
    _, _, transition_starts, transition_ends = peak_widths(change_point_signal, peaks, rel_height=0.5)
    # transition_starts = list(map(int, np.round(transition_starts)))
    #transition_ends = list(map(int, np.round(transition_ends)))

    _, _, transition_starts_conf, transition_ends_conf = peak_widths(change_point_signal, conf_peaks, rel_height=0.5)
    transition_starts = list(map(int, np.round(transition_starts_conf)))
    transition_ends = list(map(int, np.round(transition_ends_conf)))
   
    # Étape 7 : Détection des régions de notes candidates
    # get candidate note regions - any point between two peaks in the change point signal
    transitions = [(s, f, 'transition') for (s, f) in zip(transition_starts, transition_ends)]
  
    note_starts = [0] + transition_ends
    note_ends = transition_starts + [len(change_point_signal) + 1]
    note_regions = [(s, f, 'note') for (s, f) in (zip(note_starts, note_ends))]

    # Étape 8 : Détection de l'amplitude (optionnel)
    if detect_amplitude:
        # take the amplitudes within 6 sigma of the mean
        # helps to clean up outliers in amplitude scaling as we are not looking for 100% accuracy
        amp_mean = np.mean(filtered_amp_envelope)
        amp_sd = np.std(filtered_amp_envelope)
        # filtered_amp_envelope = amp_envelope.copy()
        filtered_amp_envelope[filtered_amp_envelope > amp_mean + (6 * amp_sd)] = 0
        global_max_amp = max(filtered_amp_envelope)
 

    # Étape 9 : Création de la liste des segments
    segment_list = []
    for a, b, label in sum(zip(note_regions, transitions), ()):
        if label == 'transition':
            continue

        # Handle an edge case where rounding could cause
        # an end index for a note to be before the start index
        if a > b:
            continue
        elif b - a <= 1:
            continue

        if detect_amplitude:
            max_amp = np.max(filtered_amp_envelope[a:b])
            scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))
          
        else:
            scaled_max_amp = 80
        
        # Nouvelle etape : filtrer les fréquences pour ne gardes que celle autour de la note que l'on veut détecter
        if np.round(np.median(midi_pitch[a:b]))== midi_note:
            pitch= np.round(np.median(midi_pitch[a:b]))
            # print(f"midi pitch: {pitch}")
        else:
            pitch=0
        segment_list.append({
            'pitch': pitch,
            'freq': np.median(freqs[a:b]),
            'conf': np.median(conf[a:b]),
            'transition_strength': 1 - conf[a], # TODO: make use of the dip in confidence as a measure of how strong an onset is
            'amplitude': scaled_max_amp,
            'start_idx': a,
            'finish_idx': b,
        })

    ## Étape 10 : Filtrage et fusion des segments
    # segment list contains our candidate notes
    # now we iterate through them and merge if two adjacent segments have the same median pitch
    notes = []
    sub_list = []
    for a, b in zip(segment_list, segment_list[1:]):
        # TODO: make use of variance in segment to catch glissandi?
        # if np.var(midi_pitch[a[1][0]:a[1][1]]) > 1:
        #     continue

        if np.abs(a['pitch'] -
                  b['pitch']) > 0.5:  # or a['transition_strength'] > 0.4:
            sub_list.append(a)
            notes.append(sub_list)
            sub_list = []
        else:
            sub_list.append(a)

    # catch any segments at the end
    if len(sub_list) > 0:
        notes.append(sub_list)


    # garder les notes si le pitch coorespond à la note
    # Étape 11 : Création des notes de sortie
  
    velocities = []
    durations = []
    output_notes = []
    min_diff=8
    # Filter out notes that are too short or too quiet
    for x_s in notes:
        
        #x_s_filt = [x for x in x_s if x['amplitude'] > min_velocity and abs(x['pitch']-midi_note)<min_diff]
        x_s_filt = [x for x in x_s if x['amplitude'] > min_velocity and midi_note-1<=x['pitch']<=midi_note+1]
        # print x_s_filt['amplitude']
        
        #x_s_filt = [x for x in x_s if x['pitch']==midi_note]
        #x_s_filt = [x for x in x_s if abs(x['pitch']-midi_note)<min_diff]    
       
        if len(x_s_filt) == 0:
            continue
        median_pitch = np.median(np.array([y['pitch'] for y in x_s_filt]))
        median_confidence = np.median(np.array([y['conf'] for y in x_s_filt]))
        seg_start = x_s_filt[0]['start_idx']
        seg_end = x_s_filt[-1]['finish_idx']
        time_start = 0.01 * seg_start
        time_end = 0.01 * seg_end
        sample_start = time_to_samples(time_start, sr=sr)
        sample_end = time_to_samples(time_end, sr=sr)
        max_amp = np.max(filtered_amp_envelope[seg_start:seg_end])
        scaled_max_amp = np.interp(max_amp, (0, global_max_amp), (0, 127))

        valid_amplitude = scaled_max_amp > min_velocity
        valid_duration = (time_end - time_start) > min_duration
        
        # TODO: make use of confidence strength
        valid_confidence = True  # median_confidence > 0.1

        #if valid_amplitude and valid_confidence and valid_duration:
        output_notes.append({
            'pitch':
                int(np.round(median_pitch)),
            'freq':
                x_s_filt[0]['freq'],
            'velocity':
                round(scaled_max_amp),
            'start_idx':
                seg_start,
            'finish_idx':
                seg_end,
            'conf':
                median_confidence,
            'transition_strength':
                x_s[-1]['transition_strength']
        })
   
    # Étape 13 : Gestion des notes répétées 
    # Handle repeated notes
    # Here we use a standard onset detection algorithm from madmom
    # with a high threshold (0.8) to re-split notes that are repeated
    # Repeated notes have a pitch gradient of 0 and are therefore
    # not separated by the algorithm above
    if not disable_splitting:
        onset_separated_notes = []
        for n in output_notes:
            n_s = n['start_idx']
            n_f = n['finish_idx']

            last_onset = 0
            if np.any(onsets[n_s:n_f] > 0.7):
                onset_idxs_within_note = np.argwhere(onsets[n_s:n_f] > 0.7)
                for idx in onset_idxs_within_note:
                    if idx[0] > last_onset + int(min_duration / 0.01):
                        new_note = n.copy()
                        new_note['start_idx'] = n_s + last_onset
                        new_note['finish_idx'] = n_s + idx[0]
                        onset_separated_notes.append(new_note)
                        last_onset = idx[0]

            # If there are no valid onsets within the range
            # the following should append a copy of the original note,
            # but if there were splits at onsets then it will also clean up any tails
            # left in the sequence
            new_note = n.copy()
            new_note['start_idx'] = n_s + last_onset
            new_note['finish_idx'] = n_f
            onset_separated_notes.append(new_note)
            output_notes = onset_separated_notes
    
    if detect_amplitude:
        # Trim notes that fall below a certain amplitude threshold
        timed_output_notes = []
        for n in output_notes:
            timed_note = n.copy()

            # Adjusting the start time to meet a minimum amp threshold
            s = timed_note['start_idx']
            f = timed_note['finish_idx']

            if f - s > (min_duration / 0.01):
                # TODO: make noise floor configurable
                noise_floor = 0.01  # this will vary depending on the signal
                s_samp = steps_to_samples(s, sr)
                f_samp = steps_to_samples(f, sr)
                s_adj_samp_all = s_samp + np.where(
                    filtered_amp_envelope[s:f] > noise_floor)[0]

                if len(s_adj_samp_all) > 0:
                    s_adj_samp_idx = s_adj_samp_all[0]
                else:
                    continue

                s_adj = samples_to_steps(s_adj_samp_idx, sr)

                f_adj_samp_idx = f_samp - np.where(
                    np.flip(filtered_amp_envelope[s:f]) > noise_floor)[0][0]
                if f_adj_samp_idx > f_samp or f_adj_samp_idx < 1:
                    print("something has gone wrong")

                f_adj = samples_to_steps(f_adj_samp_idx, sr)
                if f_adj > f or f_adj < 1:
                    print("something has gone more wrong")

                timed_note['start'] = s_adj * 0.01
                timed_note['finish'] = f_adj * 0.01
                timed_output_notes.append(timed_note)
            else:
                timed_note['start'] = s * 0.01
                timed_note['finish'] = f * 0.01
        
    
    return timed_output_notes, filtered_amp_envelope