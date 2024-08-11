import json
from TTS.api import TTS
import torch
import torchaudio
import numpy as np
from transformers import pipeline
from googletrans import Translator
from pydub import AudioSegment
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
STT_MODEL_NAME = "openai/whisper-tiny"

stt_pipe = pipeline(
    task="automatic-speech-recognition",
    model=STT_MODEL_NAME,
    chunk_length_s=30,
    device=device,
)
tts_model = TTS(model_name=TTS_MODEL_NAME, progress_bar=True).to(device)
translator = Translator()


def speech_to_text(wav_path):
    # Load audio using torchaudio for better handling
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        sample_rate = 16000
    waveform = waveform.squeeze(0)
    
    # Process audio
    out = stt_pipe(waveform.numpy(), return_timestamps=True)

    chunks: list[dict] = out["chunks"]
    reformatted_chunks = []
    for chunk in chunks:
        start, end = chunk["timestamp"]
        text = chunk["text"]
        reformatted_chunks.append({"start": start, "end": end, "text": text})

    return reformatted_chunks

def translate_text(text, target_language_code):
    if text == None or len(text.strip()) == 0:
        return ''
    translation = translator.translate(text, dest=target_language_code)
    return translation.text

def text_to_speech(text, lan:str, tts:TTS, speaker_wav):
    wav_path = "temp.wav"
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=lan, file_path=wav_path)
    
    return wav_path

def create_translated_audio(wav_path, target_language):
    # Step 1: Get the speech chunks (with timestamps) from the original audio
    chunks = speech_to_text(wav_path)
    
    # Step 2: Initialize an empty audio segment to combine results
    combined_audio = AudioSegment.silent(duration=0)
    
    # Step 3: Process each chunk
    for chunk in chunks:
        start_time = chunk['start'] * 1000  # Convert to milliseconds
        end_time = chunk['end'] * 1000  # Convert to milliseconds
        text = chunk['text']
        
        # Step 4: Translate the text
        translated_text = translate_text(text, target_language)
        
        # Step 5: Generate speech for the translated text
        speaker_wav = wav_path  # Using the original speaker's voice as a reference
        translated_wav_path = text_to_speech(translated_text, target_language, tts_model, speaker_wav)
        
        # Step 6: Load the generated speech
        translated_audio_segment = AudioSegment.from_wav(translated_wav_path)
        
        # Step 7: Adjust the length of the translated speech to match the original segment
        original_duration = end_time - start_time
        translated_audio_segment = translated_audio_segment.set_frame_rate(int(original_duration / len(translated_audio_segment) * translated_audio_segment.frame_rate))
        
        # Step 8: Add the translated audio segment to the combined audio
        combined_audio += AudioSegment.silent(duration=start_time - len(combined_audio))  # Add silence until the current segment's start
        combined_audio += translated_audio_segment[:original_duration]  # Add the translated segment, trimmed to the original length
    
    # Step 9: Save the combined audio
    output_path = "translated_audio.wav"
    combined_audio.export(output_path, format="wav")

    # Delete temp files
    os.remove('temp.wav')
    
    return output_path

# Example usage
translated_audio_path = create_translated_audio("../private/test_clip.wav", "fr")
print(f"Translated audio saved to: {translated_audio_path}")
