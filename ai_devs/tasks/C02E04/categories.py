import base64
from openai import OpenAI
from icecream import ic
from dotenv import load_dotenv
import os
from pathlib import Path
from pydantic import BaseModel
from typing import Literal

import json

from utils.client import AIDevsClient

# ? Download the files, extract  (available here: https://centrala.ag3nts.org/dane/pliki_z_fabryki.zip)
# ? and save them in the tasks/C02E04/data folder.
# ! Files are not provided in the repository due to privacy reasons.

class Classification(BaseModel):
    thoughts: str
    class_prediction: Literal["people", "hardware", "other"]




# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize clients
aidevs_client = AIDevsClient(
    api_key=os.getenv("AIDEVS_API_KEY"),
    base_url="https://centrala.ag3nts.org/",
    verify_dir="report",
)


def get_files(folder_path: Path) -> list:
    return [file for file in folder_path.glob('*') if file.suffix in SUPPORTED_EXTENSIONS]

def transcribe_audio(audio_path: Path) -> str:
    try:
        with open(audio_path, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return response.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def ocr(image_path: Path):
    base64_image = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

    ocr_prompt = """
    Act as a system that processes the image and extracts the text from it.
    Respond with the extracted text just like an OCR system would.
    Do NOT include any additional information or context.
    """

    analysis = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": ocr_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":  f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ]
            }
        ],
    )

    return analysis.choices[0].message.content

def classify_information(text: str) -> Classification:

    classification_prompt = """
    Act as a system that processes text and based on the information found, classify it into one of the following categories: 'people', 'hardware' or 'other'.
    The text should be analyzed for any relevant information that would help determine the category.

    Use the following Rules to classify the text:
    <people>
    - check if the text contains about captured/suspicous/dangerous/unsual people or traces of their presence.
    - look for any descriptions of individuals or groups that are deemed suspicious or dangerous.
    - ignore any mentions of regular or normal people e.g people deemed friendly or helpful.
    </people>

    <hardware>
    - check if the text contains information about any BROKEN (physical) equipment, tools, or machinery.
    - look for any descriptions of damaged or malfunctioning hardware.
    - ignore any mentions of working or functional equipment.
    - ignore software-related (non-physical / code) issues e.g. system or protocol updates, software bugs.
    </hardware>

    <other>
    - if the text does not contain any information about people or hardware, classify it as 'other'.
    </other>

    Return the classification result in the following format:
    {
        "thoughts": "Your thoughts on the classification e.g. why you chose a specific category",
        "class_prediction": "The predicted class based on the text [people, hardware, other]"
    }
    """

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": classification_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ],
        response_format=Classification
    )
    
    return completion.choices[0].message.parsed


DATA_PATH = Path('./tasks/C02E04/data/')
TRANSCRIPTION_PATH = Path('./tasks/C02E04/transcription/')
OCR_PATH = Path('./tasks/C02E04/ocr/')
CLASSIFICATION_PATH = Path('./tasks/C02E04/classifications/')

SUPPORTED_EXTENSIONS = ['.txt', '.mp3', '.png']

files = get_files(DATA_PATH)
ic(files)

results = [
]

for file in files:
    filename = file.stem
    suffix = file.suffix

    if suffix == '.txt':
        ic(f"Loading text file {filename}")
        text = file.read_text(encoding="utf-8")

    elif suffix == '.mp3':
        ic(f"Processing audio file {filename}")
        # Check if the transcription file already exists
        transcription = TRANSCRIPTION_PATH / f"{filename}.txt"

        if transcription.exists():
            ic(f"Loading existing transcription for {filename}")
            text = transcription.read_text(encoding="utf-8")
        else:
            ic(f"Transcribing audio file {filename}")
            text = transcribe_audio(file)

            # Save transcription to file
            with transcription.open("w", encoding="utf-8") as f:
                f.write(text)
    
    elif suffix == '.png':
        ic(f"Processing image file {filename}")
        # Check if the OCR file already exists
        ocr_file = OCR_PATH / f"{filename}.txt"

        if ocr_file.exists():
            ic(f"Loading existing OCR result for {filename}")
            text = ocr_file.read_text(encoding="utf-8")
        else:
            ic(f"Performing OCR on {filename}")
            text = ocr(file)

            # Save OCR result to file
            with ocr_file.open("w", encoding="utf-8") as f:
                f.write(text)

    ic(text)

    # Check if the classification file already exists
    classification_file = CLASSIFICATION_PATH / f"{filename}.json"

    if classification_file.exists():
        ic(f"Loading existing classification for {filename}")

        with open(classification_file, 'r', encoding="utf-8") as f:
            json_data= json.load(f)
            classification = Classification.model_validate(json_data)
    else:
        ic(f"Classifying information from {filename}")
        classification = classify_information(text)

        # Save classification to file
        with classification_file.open("w", encoding="utf-8") as f:
            f.write(classification.model_dump_json())

    # Add to results
    results.append({
        "file": file,
        "classification": classification
    })


for result in results:
    ic(result)

# Extract only the "people" and "hardware" classifications and order them by filename
solution = {
    "people": sorted([x['file'].name for x in results if x["classification"].class_prediction == "people"]),
    "hardware": sorted([x['file'].name for x in results if x["classification"].class_prediction == "hardware"])
}

ic(solution)


# Submit the generated image to the AIDevs API
response = aidevs_client.verify_task("kategorie", solution)
ic(response)

