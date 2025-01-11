from openai import OpenAI
from icecream import ic
from dotenv import load_dotenv
import os
from pathlib import Path

# ? Download and unzip the audio files (available here: https://centrala.ag3nts.org/dane/przesluchania.zip)
# ? and save them in the tasks/C02E01/recordings folder.
# ! Recordings are not provided in the repository due to privacy reasons.

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_audio_files(folder_path):
    return [audio_file for audio_file in folder_path.glob('*.m4a')]

def save_transcription(transcription, file_path):
    with file_path.open("w", encoding='UTF-8') as file:
        file.write(transcription.text)

# Use OpenAI Whisper model to transcribe audio/speech to text
recordings_path = Path('./tasks/C02E01/recordings/')
transcriptions_path = Path('./tasks/C02E01/transcriptions/')
audio_files_paths = get_audio_files(recordings_path)

for file_path in audio_files_paths:
    with file_path.open("rb") as file:
        transcription = openai_client.audio.transcriptions.create(
        model="whisper-1", 
        file=file
        )

        ic(transcription.text)

        # Save transcriptions to file
        transcription_file_path = transcriptions_path / f"{file_path.stem}.txt"
        save_transcription(transcription, transcription_file_path)






