# Function to label overlap segments with the speakers involved
def label_overlap(overlap, diarization):
    new_overlapped_segments = []

    # Iterate over detected overlap segments
    for overlap_segment in overlap.itertracks():
        overlapped_speakers = []

        # Check for intersection with diarization segments
        for segment in diarization.itertracks(yield_label=True):
            speaker_label = segment[2]  # Extract the speaker label
            if overlap_segment[0].intersects(segment[0]):
                overlapped_speakers.append(speaker_label)

        # Add the overlap segment and corresponding speakers
        new_overlapped_segments.append(
            (overlap_segment[0], list(set(overlapped_speakers)))  # Remove duplicates
        )

    return new_overlapped_segments


# Function to clean diarization output
def clean_diarization(diarization):
    labeled_segments = []

    # Extract segments and labels from diarization
    for segment in diarization.itertracks(yield_label=True):
        labeled_segments.append((segment[0], segment[2]))

    return labeled_segments


# Function to group segments by speaker
def group_by_speaker(single_segments, overlap_segments, speakers):
    all_segments = []

    # Iterate over each speaker
    for i, speaker in enumerate(speakers):
        speaker_segments = []

        # Collect single-speaker segments for the current speaker
        for segment, label in single_segments:
            if speaker == label:
                speaker_segments.append((segment, speaker))

        # Collect overlap segments where the current speaker is involved
        for segment, labels in overlap_segments:
            if speaker in labels:
                speaker_segments.append((segment, 'overlap'))  # Label as 'overlap'

        # Order segments by start time
        ordered_speaker_segments = sorted(speaker_segments, key=lambda x: x[0].start)
        all_segments.append(ordered_speaker_segments)

    return all_segments
