# Political Debate Analysis Pipeline

> AI-based pipeline for analyzing political debates with speaker diarization, overlap detection, transcription, speaker identification, and visual playback.

## About the Project

This project is my end-of-studies master's thesis, focused on building a pipeline that analyzes political debates using artificial intelligence.

The system takes a `.wav` audio file of a political debate involving multiple speakers and processes it through several stages:

1. **Speaker Diarization**: Segment the audio into regions corresponding to different speakers.
2. **Overlap Detection**: Identify and separate segments where multiple speakers talk at the same time, which is a common occurrence during debates.
3. **Segmentation**: Group the audio segments by speaker or overlap.
4. **Transcription**: Transcribe each individual segment once overlaps are resolved.
5. **Speaker Identification**: Predict the identity of each speaker based on their audio segment.
6. **Playback Visualization**: Provide a playback of the original debate audio, displaying the speaker’s identity dynamically while they are talking.

## Important Notice

Due to confidentiality agreements with the company I collaborated with, I am not allowed to share the training data used for speaker identification.

If you wish to test the pipeline yourself, you will need to provide your own dataset.  
A notebook (`speaker_identification_model.ipynb`) is included to help you create and train your own speaker identification model, along with some sample analyses demonstrating its effectiveness.
