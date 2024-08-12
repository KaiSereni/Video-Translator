from translator_function import create_translated_audio
from moviepy.editor import VideoFileClip, AudioFileClip
import os
from translator_function import create_translated_audio

with open('output_language.txt', 'r') as file:
    lan = file.read().strip()

config = {
    "video_path": "src/input.mp4",
    "lan": lan
}

def add_audio_to_video(video_path, audio_path, output_video_path):
    video_clip = VideoFileClip(video_path)
    
    # Load the processed audio
    audio_clip = AudioFileClip(audio_path)
    
    # Set the audio of the video clip to the processed audio
    video_with_new_audio = video_clip.set_audio(audio_clip)
    
    # Write the result to a file
    video_with_new_audio.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

def main(video_path, lan):
    # Extract the audio from the video
    video_clip = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video_clip.audio.write_audiofile(audio_path)
    
    # Process the extracted audio
    processed_audio_path = create_translated_audio(audio_path, lan)
    
    # Create output video file path
    output_video_path = "output.mp4"
    
    # Add the processed audio back to the video
    add_audio_to_video(video_path, processed_audio_path, output_video_path)
    
    # Clean up temporary audio files
    os.remove(audio_path)
    os.remove(processed_audio_path)
    
    return output_video_path

output = main(config["video_path"], config["lan"])
print(f"Translated video saved to: {output}")