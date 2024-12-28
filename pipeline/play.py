from pydub import AudioSegment
from pydub.playback import play
import time
import threading

# Function to combine single speaker segments and overlap segments into a unified timeline
def combine_segments(single_segments, overlap_dict):
    all_segments = sorted(
        single_segments + [(segment, 'overlap') for segment in overlap_dict.keys()],
        key=lambda x: x[0].start  # Sort by the start time of the segment
    )
    return all_segments

# Function to print speaker labels during playback of an audio file
def print_speaker_during_playback(all_segments, speaker_path_dict, single_predictions, INPUT_PATH):
    # Load the audio file to be played
    audio = AudioSegment.from_wav(INPUT_PATH)

    # Function to play the audio in a separate thread
    def play_audio(audio):
        play(audio)
    
    # Start a new thread for audio playback to allow simultaneous segment tracking
    threading.Thread(target=play_audio, args=(audio,)).start()

    start_time = time.time()  # Record the start time of playback
    current_segment_index = 0  # Index to track the current segment
    last_speaker = None  # Keep track of the last speaker to avoid redundant prints

    # Loop until the audio playback completes
    while time.time() - start_time < len(audio) / 1000:  # Convert audio duration from ms to seconds
        segment, speaker = all_segments[current_segment_index]  # Get the current segment and its speaker
        segment_start = segment.start  # Start time of the segment
        segment_end = segment.end  # End time of the segment

        # Check if the current time falls within the segment's time range
        if segment_start <= time.time() - start_time < segment_end:
            if speaker != 'overlap':  # For single speaker segments
                speaker_path = speaker_path_dict[speaker]  # Get the speaker's audio path
                speaker_name = single_predictions[speaker_path]  # Map the path to the speaker's name
                if speaker_name != last_speaker:  # Print speaker only if it has changed
                    print(f"Speaker: {speaker_name}")
                    last_speaker = speaker_name
            else:  # For overlap segments
                if "Multiple speakers" != last_speaker:  # Avoid redundant "Multiple speakers" print
                    print("Multiple speakers")
                    last_speaker = "Multiple speakers"

        elif time.time() - start_time >= segment_end:  # Move to the next segment after the current one ends
            current_segment_index += 1

        time.sleep(0.1)  # Introduce a short delay to prevent excessive CPU usage
