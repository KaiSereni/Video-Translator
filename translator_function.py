import json
import os
from TTS.api import TTS
import torch
import torchaudio
import numpy as np
from transformers import pipeline
from googletrans import Translator
from pydub import AudioSegment, effects

# Set the device to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Specify the model names for Text-to-Speech (TTS) and Speech-to-Text (STT)
#STT_MODEL_NAME = "openai/whisper-tiny"
#STT_MODEL_NAME = "openai/whisper-large-v3"
#TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
with open('config.json') as f:
    config = json.load(f)
    STT_MODEL_NAME = config["STT_MODEL_NAME"]
    TTS_MODEL_NAME = config["TTS_MODEL_NAME"]

# Initialize the STT pipeline and TTS model
stt_pipe = pipeline(
    task="automatic-speech-recognition",
    model=STT_MODEL_NAME,
    chunk_length_s=30,
    device=device,
)
tts_model = TTS(model_name=TTS_MODEL_NAME, progress_bar=True).to(device)

# Initialize the Google Translate API
translator = Translator()

def speech_to_text(wav_path: str) -> list:
    """
    Converts speech from a WAV file to text using a speech-to-text model.

    Args:
        wav_path (str): Path to the input WAV file.

    Returns:
        list: A list of dictionaries containing the start and end timestamps 
              and the corresponding transcribed text for each chunk.
    """
    # Load the waveform and sample rate from the WAV file
    waveform, sample_rate = torchaudio.load(wav_path)

    # Resample the audio to 16kHz if needed (common sample rate for STT models)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        
    # Flatten the waveform to ([160000])
    waveform = waveform.mean(dim=0, keepdim=True)
    
    # Flatten the waveform tensor to 1D
    waveform = waveform.squeeze(0)
    
    # Get the transcriptions with timestamps
    out = stt_pipe(waveform.numpy(), return_timestamps=True)

    # Reformat the chunks for easier processing later
    reformatted_chunks = [
        {"start": chunk["timestamp"][0], "end": chunk["timestamp"][1], "text": chunk["text"]}
        for chunk in out["chunks"]
    ]

    return reformatted_chunks

def translate_text(text: str, target_language_code: str) -> str:
    """
    Translates text into a target language using Google Translate API.

    Args:
        text (str): The input text to translate.
        target_language_code (str): The language code for the target language.

    Returns:
        str: The translated text.
    """
    if not text.strip():
        return ''
    translation = translator.translate(text, dest=target_language_code)
    return translation.text

def cap_audio_length(audio_path: str, max_duration_sec: int = 6) -> str:
    """
    Caps the duration of an audio file to a maximum duration by trimming or padding.

    Args:
        audio_path (str): Path to the input audio file.
        max_duration_sec (int): Maximum duration in seconds.

    Returns:
        str: Path to the capped audio file.
    """
    audio = AudioSegment.from_wav(audio_path)
    max_duration_ms = max_duration_sec * 1000  # Convert to milliseconds
    
    if len(audio) > max_duration_ms:
        # Trim the audio to the max duration
        capped_audio = audio[:max_duration_ms]
    else:
        # Pad the audio with silence to reach the max duration
        silence_padding = AudioSegment.silent(duration=max_duration_ms - len(audio))
        capped_audio = audio + silence_padding
    
    capped_audio_path = "capped_speaker.wav"
    capped_audio.export(capped_audio_path, format="wav")
    return capped_audio_path

def text_to_speech(text: str, language: str, tts: TTS, speaker_wav: str) -> str:
    """
    Converts text to speech and saves it to a WAV file.

    Args:
        text (str): The input text to convert.
        language (str): The language code for the speech synthesis.
        tts (TTS): The TTS model instance.
        speaker_wav (str): Path to the speaker's WAV file.

    Returns:
        str: The path to the generated WAV file.
    """
    # Cap the speaker_wav to a maximum of 6 seconds
    capped_speaker_wav = cap_audio_length(speaker_wav, max_duration_sec=6)
    
    wav_path = "temp.wav"
    tts.tts_to_file(text=text, speaker_wav=capped_speaker_wav, language=language, file_path=wav_path)
    
    # Remove the temporary capped speaker wav file
    os.remove(capped_speaker_wav)
    
    return wav_path

def match_duration_and_remove_silence(audio_segment: AudioSegment, target_duration_ms: int) -> AudioSegment:
    """
    Matches the duration of an audio segment to the target duration and removes silence.

    Args:
        audio_segment (AudioSegment): The audio segment to process.
        target_duration_ms (int): The target duration in milliseconds.

    Returns:
        AudioSegment: The processed audio segment with matched duration and no silence.
    """
    # Remove silence from the audio
    non_silent_audio = effects.strip_silence(audio_segment, silence_len=100, silence_thresh=-50)
    
    # Match the duration by speeding up or slowing down the audio
    speed_factor = len(non_silent_audio) / target_duration_ms
    if speed_factor > 1:
        non_silent_audio = non_silent_audio.speedup(playback_speed=speed_factor)
    else:
        non_silent_audio = non_silent_audio.set_frame_rate(int(non_silent_audio.frame_rate * speed_factor))
    
    return non_silent_audio

def create_translated_audio(wav_path: str, target_language: str) -> str:
    """
    Creates a translated version of the input audio by transcribing, translating, 
    and then synthesizing the speech back into the target language.

    Args:
        wav_path (str): Path to the input WAV file.
        target_language (str): The target language code for translation.

    Returns:
        str: The path to the final translated audio WAV file.
    """
    # Get the transcribed chunks from the original audio
    chunks = speech_to_text(wav_path)
    
    # Create an empty audio segment to store the final audio
    combined_audio = AudioSegment.silent(duration=0)
    
    # Process each transcribed chunk
    for chunk in chunks:
        start_time = chunk['start'] * 1000  # Convert seconds to milliseconds
        end_time = chunk['end'] * 1000
        text = chunk['text']
        
        # Translate the text into the target language
        translated_text = translate_text(text, target_language)
        
        # Generate speech from the translated text
        speaker_wav = wav_path
        translated_wav_path = text_to_speech(translated_text, target_language, tts_model, speaker_wav)
        
        # Load the translated audio segment
        translated_audio_segment = AudioSegment.from_wav(translated_wav_path)
        
        # Match the duration of the translated segment with the original and remove silence
        original_duration = end_time - start_time
        translated_audio_segment = match_duration_and_remove_silence(translated_audio_segment, original_duration)
        
        # Add silence before the current translated segment if necessary
        combined_audio += AudioSegment.silent(duration=start_time - len(combined_audio))
        
        # Append the translated segment, trimming to match the original duration
        combined_audio += translated_audio_segment[:original_duration]
    
    # Export the combined translated audio to a file
    output_path = "translated_audio.wav"
    combined_audio.export(output_path, format="wav")

    # Clean up the temporary file
    os.remove('temp.wav')
    
    return output_path