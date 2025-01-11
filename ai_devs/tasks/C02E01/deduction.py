from dataclasses import dataclass
from openai import OpenAI
from icecream import ic
from dotenv import load_dotenv
import os
from pathlib import Path

from utils.client import AIDevsClient
import json

from pydantic import BaseModel

CONTEXT_TEMPLATE = "-------------------------\n### Interview: {person}\n\n{summary}\n-------------------------"
TRANSCRIPTIONS_PATH = Path("./tasks/C02E01/transcriptions/")
SUMMARIES_PATH = Path("./tasks/C02E01/summaries/")

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


# Pydantic model used for the GTP-4o structured output
# ? https://platform.openai.com/docs/guides/structured-outputs
class Deduction(BaseModel):
    thoughts: list[str]
    street: str


def extract_information(transcription: str) -> str:
    # Extract information from the transcription that suggests the location of the department / institute / faculty

    extraction_prompt = Path("./tasks/C02E01/system_prompts/extraction.txt").read_text()

    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": transcription},
        ],
    )

    return completion.choices[0].message.content


def save_summary(summary: str, file_path: Path) -> None:
    with file_path.open("w", encoding="UTF-8") as file:
        file.write(summary)


def main():
    transcription_file_paths = [file for file in TRANSCRIPTIONS_PATH.glob("*.txt")]

    # Process each transcription file
    context_parts = []
    for file_path in transcription_file_paths:
        with file_path.open("r", encoding="UTF-8") as file:
            transcription = file.read()

            # Load the summary from the file if it exists, otherwise extract the information and save it
            summary_file_path = SUMMARIES_PATH / f"{file_path.stem}.txt"

            if summary_file_path.exists():
                summary = summary_file_path.open("r", encoding="UTF-8").read()
            else:
                summary = extract_information(transcription)
                save_summary(summary, summary_file_path)

            # Create the context entry for the response
            context_part = CONTEXT_TEMPLATE.format(
                person=file_path.stem.capitalize(), summary=summary
            )

            # Append the context entry to the list of context parts
            context_parts.append(context_part)

    # Combine all context parts into a single context string that includes the name of the interviewee and the summary
    context = "\n\n".join(context_parts)

    # Load the deduction prompt from the file
    deduction_prompt = Path("./tasks/C02E01/system_prompts/deduction.txt").read_text()

    # Use Structured Outputs beta to ensure the response is in the correct format
    # ! Prompt / Completion solution is not always correct 
    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": deduction_prompt},
            {"role": "user", "content": context},
        ],
        response_format=Deduction,
    )

    # Extract and print the response from the completion
    data = completion.choices[0].message.content
    ic(data)

    # Extract the street name from the response
    json_data = json.loads(data)

    # Verify the task
    response = aidevs_client.verify_task("mp3", json_data["street"])
    ic(response)


if __name__ == "__main__":
    main()
