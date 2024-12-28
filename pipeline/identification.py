import numpy as np
from pyannote.audio import Model
from pyannote.audio import Inference
import tensorflow as tf
import os
from pydub import AudioSegment

MODEL = tf.keras.models.load_model(r'model.h5')
SPEAKER_DIC = {0: 'Hollande', 1: 'Macron', 2: 'Attal', 3: 'Hamon', 4: 'Mitterrand', 5: 'MLP', 6: 'Royal', 7: 'Schiappa', 8: 'Zemmour',
               9: 'Pecresse', 10: 'Giscard', 11: 'JLM', 12: 'Sarkozy', 13: 'Fillon', 14: 'Darmanin', 15: 'Veran', 16: 'Bardella', 17: 'Autain'}


def identify_speaker(folder, model, speaker_dic,allowed_speakers=None):
    files = os.listdir(folder)
    files_list = [os.path.normpath(os.path.join(folder, path)) for path in files]
    predictions = {}
    for path in files_list:
        print(path)
        pipeline = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=os.getenv('HF_TOKEN'))
        inference = Inference(pipeline, window="whole")
        embedded = inference(path)
        pred = model.predict(embedded.reshape(1, 512, 1))
        if allowed_speakers is not None:
            disallowed_speakers = [k for k, v in speaker_dic.items() if v not in allowed_speakers]
            pred[0, disallowed_speakers] = 0
        speaker = np.argmax(pred)
        predictions[path] = speaker_dic[speaker]
    return predictions

def single_speaker_dict(predictions, speaker_dict):
    label_dict = {}
    for path_pred, pred in predictions.items():
        filename_pred = os.path.splitext(os.path.basename(path_pred))[0]
        for speaker, path in speaker_dict.items():
            filename = os.path.splitext(os.path.basename(path))[0]
            if filename == filename_pred:
                label_dict[speaker] = pred

    return label_dict
