from pydub import AudioSegment
import audiosegment
import os
# Uncomment the following line if you need to specify the path to FFmpeg manually
# AudioSegment.converter = "C:/Program Files/FFmpeg/ffmpeg/bin"

# Function to concatenate a list of audio clips into a single audio file
def concatenate_clips(clips):
    concat = AudioSegment.empty()  # Start with an empty AudioSegment
    for item in clips:
        concat += item  # Add each clip to the concatenated audio
    return concat

# Function to export overlapping audio segments as separate files
def export_overlap(input_audio_path, overlap_segments):
    # Load the audio file and resample it to 8kHz mono for processing
    audio = audiosegment.from_file(input_audio_path).resample(
        sample_rate_Hz=8000, channels=1)

    overlap_dict = {}  # Dictionary to map segments to file names
    for i, segment in enumerate(overlap_segments):
        overlap = segment[0]  # Extract the time range of the overlap
        start = overlap.start * 1000  # Convert start time to milliseconds
        end = overlap.end * 1000  # Convert end time to milliseconds

        clip = audio[start:end]  # Slice the audio to extract the segment
        clip.export(f'./audio/overlap/overlap_{i}.wav', format='wav')  # Export the segment
        overlap_dict[overlap] = f'overlap_{i}.wav'  # Map overlap to file name

    return overlap_dict

# Function to export monologue segments for each speaker
def export_monologue(input_audio_path, segments):
    audio = AudioSegment.from_wav(input_audio_path)  # Load the input audio file

    for i, speaker_segments in enumerate(segments):
        speaker_clips = []  # Clips for the current speaker

        # Process each segment for the current speaker
        for j, segment in enumerate(speaker_segments):
            start = segment[0].start * 1000  # Start time in milliseconds
            end = segment[0].end * 1000  # End time in milliseconds
            label = segment[1]  # Segment label (e.g., speaker name)

            if label != 'overlap':  # Ignore overlap segments
                clip = audio[start:end]  # Extract the segment
                speaker_clips.append(clip)  # Add it to the speaker's clips

        # Concatenate all clips for the speaker into a single file
        concatenation = concatenate_clips(speaker_clips)
        if concatenation:  # If there is valid audio to export
            concatenation.export(
                f"./audio/single/monologue_{i}.wav", format="wav")  # Export concatenated file


# Function to prepare monologue segments for further processing
def cut_monologue(speakers):
    folder = './audio/single'  # Folder containing monologue files
    speaker_dict = {}  # Dictionary to map speakers to their audio files

    # Iterate over all files in the folder
    for i in range(len(os.listdir(folder))):
        file = f'./audio/single/monologue_{i}.wav'  # Path to the current file
        audio = audiosegment.from_file(file).resample(
            sample_rate_Hz=16000)  # Resample the audio to 16kHz
        save_file = os.path.normpath(os.path.join('./audio/predict', f'monologue_{i}.wav'))  # Output file path

        # If the audio duration exceeds 10 seconds, cut it to 10 seconds
        if audio.duration_seconds > 10:
            clip = audio[:10000]  # Take the first 10 seconds
            clip.export(save_file, format='wav')  # Export the clipped audio
        else:
            audio.export(save_file, format='wav')  # Export the full audio if shorter than 10 seconds

        speaker_dict[speakers[i]] = save_file  # Map the speaker to their file

    return speaker_dict
