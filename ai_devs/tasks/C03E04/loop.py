import os
from dotenv import load_dotenv
from icecream import ic
from openai import OpenAI
from pydantic import BaseModel
from utils.client import AIDevsClient, AIDevsResponse
from pathlib import Path

from collections import defaultdict


class NamesPlacesResponse(BaseModel):
    names: list[str]
    places: list[str]


# Load environment variables from .env file
load_dotenv()


# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
aidevs_client = AIDevsClient(
    api_key=os.getenv("AIDEVS_API_KEY"),
    base_url="https://centrala.ag3nts.org/",
    verify_dir="report",
)



TASK_NAME = "loop"
FILE_PATH = Path("./tasks/C03E04/barbara.txt")


def extract_names_and_places(text: str) -> NamesPlacesResponse:

    system_prompt = """
    Your task is to extract Polish first names and city names from the given text.
    Names are proper nouns that refer to a specific person or city.

    <RULES>
    1. Extract only the first name and transform it to it's nominative form e.g. "Janowi" -> "Jan".
    2. Extract city names and transform them to their nominative form e.g. "Warszawie" -> "Warszawa".
    3. Make sure to handle Polish diacritics properly and to check if the extracted names are valid e.g. no misspelled names.
    4. Return the extracted names and places as lists of strings following the format in the <FORMAT> section.
    </RULES>

    <FORMAT>
    {
        "names": ["Jan", "Anna"],
        "places": ["Warszawa", "Krakow"]
    }
    </FORMAT>
    """

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
    ],
        response_format=NamesPlacesResponse,
    )

    return completion.choices[0].message.parsed

def polish_to_english_upper(text: str) -> str:
    polish_chars = 'ĄĆĘŁŃÓŚŹŻ'
    english_chars = 'ACELNOSZZ'
    trans_table = str.maketrans(polish_chars, english_chars)
    
    return text.upper().translate(trans_table)




def query_places(place: str) -> AIDevsResponse:
    response = aidevs_client.query("places", {"query": place})
    return response

def query_people(name: str) -> AIDevsResponse:
    response = aidevs_client.query("people", {"query": name})
    return response


def find_all_names_and_places(names: list[str], places: list[str]) -> dict:

    if not names and not places:
        raise ValueError("Requires at least one name or place as a starting point.")
    
    places_to_query = set(map(polish_to_english_upper, places))
    names_to_query = set(map(polish_to_english_upper, names))

    places_checked = set()
    names_checked = set()

    names_places_lkp = defaultdict(set)

    ic("Starting lookup:", names_places_lkp)

    i = 0
    while places_to_query or names_to_query:
        i += 1
        ic(f"Loop {i}")

        if places_to_query:
            # pop a place from the set
            place = places_to_query.pop()
            # mark it as checked
            places_checked.add(place)

            # query what people are associated with the place
            ic("Querying place:", place)
            try:
                response = query_places(place)
            except Exception as e:
                ic(e)
                continue

            if response.message == "[**RESTRICTED DATA**]":
                continue
            else:
                # split the response into a list of people
                new_people = response.message.split(" ")

                ic("Related people:", new_people)

                # add the place to the lookup for each person
                # and add the person to the names to query if they haven't been checked yet
                for person in new_people:
                    new_person = polish_to_english_upper(person)
                    if new_person not in names_checked:
                        names_to_query.add(new_person)
                    names_places_lkp[new_person].add(place)

        if names_to_query:
            # pop a name from the set
            name = names_to_query.pop()
            # mark it as checked
            names_checked.add(name)

            # query what places are associated with the name
            ic("Querying name:", name)
            try:
                response = query_people(name)
            except Exception as e:
                ic(e)
                continue

            if response.message == "[**RESTRICTED DATA**]":
                continue
            else:
                # split the response into a list of places
                new_places = response.message.split(" ")

                ic("Related places:", new_places)

                # add the name to the lookup for each place
                # and add the place to the places to query if they haven't been checked yet
                for place in new_places:
                    new_place = polish_to_english_upper(place)
                    if new_place not in places_checked: 
                        places_to_query.add(new_place)
                    names_places_lkp[name].add(new_place)

        ic("Current lookup:", names_places_lkp)

    return names_places_lkp


if __name__ == "__main__":
    # This tasked can be solved either by:
    # 1. Using a loop to iterate over a list of people and places with an LLM
    # 2. Be letting the LLM decide how to solve the task and which actions/requests to make

    # The first approach is more structured and deterministic, while the second approach is more flexible and open-ended.

    # Load inital text to extract names and places from
    text = FILE_PATH.read_text(encoding="UTF-8")

    # Extract names and places from the text
    response = extract_names_and_places(text)

    ic(response)

    # Find which people are associated with which places
    results = find_all_names_and_places(response.names, response.places)

    
    cities_barbara_visited = list(filter(lambda x: x != "KRAKOW", results["BARBARA"]))

    ic(cities_barbara_visited)

    # Verify the task
    for city in cities_barbara_visited:
        response = aidevs_client.verify_task(TASK_NAME, city)
        ic(response)