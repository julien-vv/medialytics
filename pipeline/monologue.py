from pyannote.core import Segment

# Function to compute the intersection of multiple segments
def intersection_segment(segments):
    intersection = segments[0]  # Start with the first segment
    for i in range(len(segments)):
        intersection = intersection & segments[i]  # Update intersection
    return intersection

# Function to extract single-speaker segments from a diarization segment
def single_speaker(segment, overlap_segments):
    relative_single_segments = []  # Store single-speaker segments
    intersected_segments = []  # Store overlapping segments

    # Iterate over overlapping segments
    for labeled_over_segment in overlap_segments:
        over_segment = labeled_over_segment[0]  # Extract overlap segment
        intersection = segment & over_segment  # Find intersection
        if intersection:  # If an intersection exists
            intersected_segments.append(over_segment)  # Track intersected segments

            # Handle parts of the segment that occur before and after the overlap
            if segment < over_segment:  # If current segment starts before overlap
                pre_overlap_segment = Segment(segment.start, over_segment.start)
                relative_single_segments.append(pre_overlap_segment)

                # Handle portion after overlap ends
                if segment.end > over_segment.end:
                    post_overlap_segment = Segment(over_segment.end, segment.end)
                    relative_single_segments.append(post_overlap_segment)
            else:  # If current segment overlaps or starts with overlap
                post_overlap_segment = Segment(over_segment.end, segment.end)
                relative_single_segments.append(post_overlap_segment)

    # If there were overlapping segments, return the isolated single-speaker portion
    if intersected_segments:
        return intersection_segment(relative_single_segments)
    else:
        # If no overlap, return the segment itself
        return segment

# Function to generate single-speaker segments from diarization and overlap data
def single_speaker_segments(diarization_segments, overlap_segments):
    single_segments = []  # Store all single-speaker segments

    # Iterate over diarization segments
    for labeled_diar_segment in diarization_segments:
        segment = labeled_diar_segment[0]  # Extract segment time range
        speaker_label = labeled_diar_segment[1]  # Extract speaker label
        single_segment = single_speaker(segment, overlap_segments)  # Isolate single-speaker portion
        single_segments.append((single_segment, speaker_label))  # Append to results

    return single_segments

# Function to extract unique speaker labels from single-speaker segments
def get_single_speakers(single_segments):
    speakers = []  # Store unique speaker labels

    # Iterate over segments and collect unique speaker labels
    for segment in single_segments:
        if segment[1] not in speakers:
            speakers.append(segment[1])

    return speakers
