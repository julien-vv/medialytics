import numpy 
import soundfile as sf 
import os 
from modelscope.pipelines import pipeline 
from modelscope.utils.constant import Tasks 

# Function to perform voice separation on audio files in a given folder
def voice_separation(input_folder, output_folder):
    files = os.listdir(input_folder)  # List all files in the input folder

    # Initialize the speech separation pipeline using the specified model
    separation = pipeline(
        Tasks.speech_separation,  # Task type: Speech Separation
        model='damo/speech_mossformer2_separation_temporal_8k'  # Pre-trained model for separation
    )

    # Iterate over each file in the input folder
    for f in files:
        # Process the file through the separation pipeline
        result = separation(os.path.join(input_folder, f))
        
        # Extract the filename without extension for output naming
        filename = os.path.splitext(f)[0]

        # Iterate over the separated audio signals
        for i, signal in enumerate(result['output_pcm_list']):
            # Define the output file path for each separated signal
            save_file = f'{output_folder}/{filename}_output_{i}.wav'

            # Save the separated audio signal to a WAV file
            sf.write(
                save_file,  # Output file path
                numpy.frombuffer(signal, dtype=numpy.int16),  # Convert PCM buffer to a NumPy array
                8000  # Sample rate in Hz
            )
