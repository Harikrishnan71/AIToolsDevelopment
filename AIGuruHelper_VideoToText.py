#pip install openai-whisper
# Import the whisper library
import whisper
video_path = "/content/Claude101.mp4"
# Load the desired Whisper model
# You can choose from 'tiny', 'base', 'small', 'medium', 'large'
# 'base' is a good balance for speed and accuracy for most cases.
print("Loading Whisper model...")

model = whisper.load_model("large")
print("Large model loaded. Now re-running transcription...")

# Transcribe the audio from the video file again with the new model
result = model.transcribe(video_path)
print("Transcription complete with medium model.")

# Display the full transcript again
print("--- Full Transcript (Medium Model) ---")
print(result["text"])
print("Model loaded.")



# Transcribe the audio from the video file

print("Transcription complete.")

print("Number of transcription segments:", len(result["segments"]))

# Save the new transcription to a text file
output_filename_medium = "transcript_medium.txt"
with open(output_filename_medium, "w") as f:
    f.write(result["text"])

print(f"New transcription successfully saved to {output_filename_medium}")
