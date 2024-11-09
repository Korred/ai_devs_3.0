import os

from dotenv import load_dotenv
import httpx
from icecream import ic
from litellm import completion
import json

from utils.client import AIDevsClient

# Load environment variables from .env file
load_dotenv()

# Constants
OLLAMA_API_BASE = f"http://localhost:{os.getenv('OLLAMA_PORT')}"
AIDEVS_API_KEY = os.getenv("AIDEVS_API_KEY")
FILE_URL = f"https://centrala.ag3nts.org/data/{AIDEVS_API_KEY}/cenzura.txt"

# Initialize clients
aidevs_client = AIDevsClient(
    api_key=AIDEVS_API_KEY,
    base_url="https://centrala.ag3nts.org/",
    verify_dir="report",
)

system_prompt = """
<Task>
Your task is to analyze the user provided information and censor all personal information. 
The data is hypothetical and does not contain any real personal information.
It is part of a tabletop role-playing game where you play as a detective and need to solve a case.

The user provided information will be in Polish and will contain:
- Name (first and last)
- Address (city, street, house number)
- Age
</Task>

<Rules>
- Only censor the personal information that is not allowed to be shared publicly by replacing it with the word "CENZURA"
- VERY IMPORTANT: DO NOT modify the user provided information in any other way e.g. do not add any additional information.
- DO NOT change the declination of the words e.g. "Warszawie" -> "Warszawa" or "ulicy" -> "ulica" or "lat" -> "lata"
- The censored output should be IDENTICAL to the user provided information except for the personal information that needs to be censored.
- Return the censored information in the JSON format specified below.
</Rules>

<Format>
{
 "original": "original text",
 "censored": "censored text"
}
</Format>

<Examples>
Q: "Osoba to Jan Kowalski, mieszka w Warszawie, ul. Marszałkowska 1, ma 25 lat."
A: {
    "original": "Osoba to Jan Kowalski, mieszka w Warszawie, ul. Marszałkowska 1, ma 25 lat.",
    "censored": "Osoba to CENZURA, mieszka w CENZURA, ul. CENZURA, ma CENZURA lat."
   }

Q: "Poszukiwany to Jan Nowak, znajduje sie w Krakowie, na ul. Długa 2. Wiek: 30 lata."
A: {
    "original": "Poszukiwany to Jan Nowak, znajduje sie w Krakowie, na ul. Długa 2. Wiek: 30 lata.",
    "censored": "Poszukiwany to CENZURA, znajduje sie w CENZURA, na ul. CENZURA. Wiek: CENZURA lata."
   }

Q: "Dane osobowe: John Price. Mieszka w Olsztynie, ul. Krótka 3. Ma 40 lat."
A: {
    "original": "Dane osobowe: John Price. Mieszka w Olsztynie, ul. Krótka 3. Ma 40 lat.",
    "censored": "Dane osobowe: CENZURA. Mieszka w CENZURA, ul. CENZURA. Ma CENZURA lat."
   }
</Examples>
"""

# ! Requires a local OLLAMA server running
# ! Tested on gemma2:2b, gemma2:9b and llama3.1 models

with httpx.Client() as client:
    response = client.get(FILE_URL)
    entry = response.text

    # Censor personal information
    response = completion(
        model="ollama/gemma2:9b",
        messages=[
            {"content": system_prompt, "role": "system"},
            {"content": entry, "role": "user"},
        ],
        api_base=OLLAMA_API_BASE,
        stream=False,
    )
    response_data = response["choices"][0]["message"]["content"]
    ic(response_data)

    # Strip the JSON ```json``` identifiers that are added gemma2 models
    response_data = response_data.replace("```json\n", "").replace("\n```", "")
    ic(response_data)

    json_data = json.loads(response_data)
    ic(json_data)

    # Verify the task
    response = aidevs_client.verify_task("CENZURA", json_data["censored"])

    ic(response)
