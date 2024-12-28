import speech_recognition as sr
import os


def transcribe(audio_path):
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    output_folder = './transcription'
    filename = os.path.splitext(os.path.basename(audio_path))[0]

    # Load the audio file
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)  # lire le fichier

    try:
        # Use the Google Web Speech API to transcribe the audio
        transcript = recognizer.recognize_google(audio, language="fr-FR")

        with open(os.path.join(output_folder, f'transcription_{filename}.txt'), 'w', encoding='latin1') as f:
            f.write(transcript)

    except sr.UnknownValueError:
        print("Google Speech Recognition n'a pas pu comprendre l'audio")
    except sr.RequestError as e:
        print(
            f"Impossible d'obtenir des résultats depuis Google Speech Recognition service; {e}")

# Function to transcribe all audio files in a given folder
def transcribe_audio(input_folder):
    files = os.listdir(input_folder)
    for file in files:
        transcribe(os.path.join(input_folder, file))
