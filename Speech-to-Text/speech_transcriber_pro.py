import speech_recognition as sr
import urllib.request
import os

def download_sample_audio():
    """Downloads a standard test audio file if it doesn't already exist."""
    file_name = "harvard_test.wav"
    url = "https://raw.githubusercontent.com/realpython/python-speech-recognition/master/audio_files/harvard.wav"
    
    if not os.path.exists(file_name):
        print(f"Downloading sample audio file ({file_name})...")
        urllib.request.urlretrieve(url, file_name)
    return file_name

def transcribe_and_save(audio_path, output_path="transcription_output.txt"):
    """Transcribes audio from a WAV file and saves the text to a file."""
    recognizer = sr.Recognizer()
    
    print(f"Loading audio file: '{audio_path}'...")
    with sr.AudioFile(audio_path) as source:
        # Calibrate the recognizer to the noise level of the audio file
        print("Calibrating for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        # Record the audio data
        print("Processing audio data...")
        audio_data = recognizer.record(source)
        
        try:
            # Recognize the speech using Google's free Web Speech API
            print("Sending to Google Web Speech API for transcription...\n")
            text = recognizer.recognize_google(audio_data)
            
            print("--- Transcription Successful ---")
            print(text)
            print("--------------------------------\n")
            
            # Save the transcribed text to a file
            with open(output_path, "w") as text_file:
                text_file.write("Speech-to-Text Transcription Result:\n")
                text_file.write("-" * 40 + "\n")
                text_file.write(text + "\n")
                
            print(f"Transcription successfully saved to '{output_path}'")
            
        except sr.UnknownValueError:
            print("Error: The speech recognition engine could not understand the audio.")
        except sr.RequestError as e:
            print(f"Error: Could not request results from the recognition service; {e}")

def main():
    # 1. Ensure we have an audio file to test with
    audio_file = download_sample_audio()
    
    # 2. Transcribe the audio and output to a text file
    transcribe_and_save(audio_file)

if __name__ == "__main__":
    main()