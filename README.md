This is my final master project, a political debate analysis pipeline using AI.

The pipeline takes a .wav audio file of a political debate between various speakers and processes it.

The first step is the diarization, which segments the audio for each different speaker.
Then the overlap detection, whenever two speakers speak at the same time, which is quite common during debates, the overlapping voices segments are extracted.  
Once the audio is segmented, we label these and group the segments by speakers or overlap.
The next step is separating the overlap segment into distinct segments for each speaker in the overlap. 
Then once every audio is free of overlap, we transcribe every segment.
The final step is to predict for each speaker, his identity.
As a bonus, we have a little playback of the original audio with each speaker being displayed while talking.

Important : I am not allowed to share the training data I have, as it is privately owned by a company I collaborated with. As such, if you want to test the pipeline on your own, you will have to get your own data. 
I have put a notebook to create your own model for speaker identification, with some few analysis of my own to demonstrate how effective it is.  
