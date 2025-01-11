import os
from pathlib import Path

from dotenv import load_dotenv
from icecream import ic
from openai import OpenAI
from utils.client import AIDevsClient

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


VERIFY_DATA_PATH = Path("./tasks/C04E02/lab_data/verify.txt")


def classify_research_data(value:str) -> bool:
    completion = openai_client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:personal:research-tf:AYgu3PmV:ckpt-step-912",
        messages=[
            {"role": "system", "content": "Classify the user provided message (combination of numbers) as TRUE or FALSE. Respond ONLY with the correct label."},
            {"role": "user", "content": value},
        ],
    )

    classification = completion.choices[0].message.content

    if classification not in ["TRUE", "FALSE"]:
        raise ValueError("Invalid classification response")
    
    else: 
        return "TRUE" == classification

    return completion.choices[0].message.content
    

if __name__ == "__main__":
    # Load data to verify
    data_to_verify = {}
    with open(VERIFY_DATA_PATH, "r") as file:
        for line in file:
            identifier, data = line.strip().split("=")
            data_to_verify[identifier] = data


    valid_data = []

    # Verify data
    for identifier, data in data_to_verify.items():
        valid = classify_research_data(data)

        ic(f"{identifier} - {data}: {valid}")
        if valid:
            valid_data.append(identifier)

    ic(valid_data)


    # Send valid data to the AIDevs API
    response = aidevs_client.verify_task("research", valid_data)
    ic(response)
