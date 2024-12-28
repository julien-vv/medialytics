import os
import tensorflow as tf
from pyannote.audio import Pipeline
from montage import export_monologue, export_overlap, cut_monologue
from monologue import single_speaker_segments, get_single_speakers
from segmentation import clean_diarization, label_overlap, group_by_speaker
from separation import voice_separation
from identification import identify_speaker, single_speaker_dict
from play import combine_segments, print_speaker_during_playback
from transcription import transcribe_audio

print("A")
# -------------------------
# CONFIGURATION & CONSTANTS
# -------------------------
SPEAKER_DIC = {
    0: 'Hollande', 1: 'Macron', 2: 'Attal', 3: 'Hamon', 4: 'Mitterrand',
    5: 'MLP', 6: 'Royal', 7: 'Schiappa', 8: 'Zemmour', 9: 'Pecresse',
    10: 'Giscard', 11: 'JLM', 12: 'Sarkozy', 13: 'Fillon', 14: 'Darmanin',
    15: 'Veran', 16: 'Bardella', 17: 'Autain'
}

INPUT_PATH = './audio/debate/zemmour_pecresse_50s.wav'
OVERLAP_FOLDER = './audio/overlap'
SINGLE_FOLDER = './audio/predict'
SEPARATION_FOLDER = './audio/separation'
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
OVERLAP_MODEL = "pyannote/overlapped-speech-detection"
IDENTIFICATION_MODEL_PATH = r'model.h5'

# -------------------------
# INITIALIZE PIPELINES
# -------------------------
def initialize_pipelines():
    diarization_pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL, use_auth_token=os.getenv('HF_TOKEN'))
    overlap_pipeline = Pipeline.from_pretrained(
        OVERLAP_MODEL, use_auth_token=os.getenv('HF_TOKEN'))
    return diarization_pipeline, overlap_pipeline


def load_identification_model(model_path):
    return tf.keras.models.load_model(model_path)


# -------------------------
# MAIN FUNCTIONS
# -------------------------
def run_pipeline():
    # Initialize models and pipelines
    diarization_pipeline, overlap_pipeline = initialize_pipelines()
    model = load_identification_model(IDENTIFICATION_MODEL_PATH)

    # Perform diarization and overlap detection
    diarization = diarization_pipeline(INPUT_PATH)
    print('Diarization performed.')
    overlap = overlap_pipeline(INPUT_PATH)
    print('Overlap performed.')

    # Process segments and speakers
    labeled_overlapped_segments = label_overlap(overlap, diarization)
    diarization_cleaned = clean_diarization(diarization)
    single_segments = single_speaker_segments(
        diarization_cleaned, labeled_overlapped_segments)
    speakers = diarization.labels()

    # Group segments and process single speakers
    all_segments = group_by_speaker(
        single_segments, labeled_overlapped_segments, speakers)
    single_speakers = get_single_speakers(single_segments)

    # Export and process audio segments
    overlap_dict = export_overlap(INPUT_PATH, labeled_overlapped_segments)
    export_monologue(INPUT_PATH, all_segments)
    voice_separation(OVERLAP_FOLDER, SEPARATION_FOLDER)

    # Transcribe audio
    transcribe_audio(SINGLE_FOLDER)
    transcribe_audio(SEPARATION_FOLDER)

    # Predict speakers
    single_predictions = identify_speaker(SINGLE_FOLDER, model, SPEAKER_DIC)
    allowed_speakers = list(set(single_predictions.values()))
    overlap_predictions = identify_speaker(
        SEPARATION_FOLDER, model, SPEAKER_DIC, allowed_speakers)

    # Display results
    print("Single Speaker Predictions:", single_predictions)
    print("Overlap Predictions:", overlap_predictions)

    # Prepare for playback
    speaker_path_dict = cut_monologue(single_speakers)
    single_speaker_dic = single_speaker_dict(single_predictions, speaker_path_dict)
    print("labels", single_speaker_dic)
    all_segments = combine_segments(single_segments, overlap_dict)
    print_speaker_during_playback(
        all_segments, speaker_path_dict, single_predictions, INPUT_PATH)

    print("Pipeline completed successfully!")


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"An error occurred: {e}")
