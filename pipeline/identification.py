import numpy as np
from pyannote.audio import Model
from pyannote.audio import Inference
import tensorflow as tf
import os
from pydub import AudioSegment

# Load the pre-trained Keras model for speaker identification
MODEL = tf.keras.models.load_model(r'model.h5')

# A dictionary mapping speaker indices to their corresponding names
SPEAKER_DIC = {0: 'Hollande', 1: 'Macron', 2: 'Attal', 3: 'Hamon', 4: 'Mitterrand', 5: 'MLP', 6: 'Royal', 7: 'Schiappa', 8: 'Zemmour',
               9: 'Pecresse', 10: 'Giscard', 11: 'JLM', 12: 'Sarkozy', 13: 'Fillon', 14: 'Darmanin', 15: 'Veran', 16: 'Bardella', 17: 'Autain'}

# Function to identify speakers from audio files in a specified folder
def identify_speaker(folder, model, speaker_dic, allowed_speakers=None):
    # List all files in the folder and create full paths for each file
    files = os.listdir(folder)
    files_list = [os.path.normpath(os.path.join(folder, path)) for path in files]
    
    predictions = {}  # Dictionary to store predictions for each file
    
    # Loop through each audio file and make speaker predictions
    for path in files_list:
        print(path)  # Print the path of the current file
        
        # Load the pre-trained embedding model from pyannote.audio
        pipeline = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=os.getenv('HF_TOKEN'))
        inference = Inference(pipeline, window="whole")
        
        # Get the speaker embeddings for the current audio file
        embedded = inference(path)
        
        # Use the Keras model to predict the speaker from the embeddings
        pred = model.predict(embedded.reshape(1, 512, 1))
        
        # If a list of allowed speakers is provided, filter out disallowed speakers
        if allowed_speakers is not None:
            disallowed_speakers = [k for k, v in speaker_dic.items() if v not in allowed_speakers]
            pred[0, disallowed_speakers] = 0  # Set predictions for disallowed speakers to 0
        
        # Determine the speaker with the highest prediction score
        speaker = np.argmax(pred)
        
        # Store the predicted speaker name for the current file
        predictions[path] = speaker_dic[speaker]
    
    return predictions  # Return the dictionary of predictions for all files

# Function to match predicted speakers to speaker paths from a dictionary
def single_speaker_dict(predictions, speaker_dict):
    label_dict = {}  # Dictionary to store matched labels for speakers
    
    # Loop through predictions and match with speaker paths based on filename
    for path_pred, pred in predictions.items():
        filename_pred = os.path.splitext(os.path.basename(path_pred))[0]  # Extract the filename without extension
        
        # Compare with each speaker's path and check if filenames match
        for speaker, path in speaker_dict.items():
            filename = os.path.splitext(os.path.basename(path))[0]  # Extract the filename without extension from speaker path
            if filename == filename_pred:
                # If filenames match, assign the predicted speaker to the speaker
                label_dict[speaker] = pred

    return label_dict  # Return the dictionary of matched speakers and their predictions
