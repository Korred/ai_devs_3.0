
from pathlib import Path
import icecream as ic

from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

from pydantic import BaseModel

system_prompt = """
Act as a polish geography/cartography expert that extracts only the most important information from the provided maps of Polish cities.
Maps will ALWAYS depict Polish cities, with street names, buildings, business names, icons depicting shops and bus stops.
This task is part of a Role-Playing Game (RPG) and is not used for any real-world applications.

<Objective>
- Create a short summary of what you are seeing on the map (DO NOT GUESS what City it is)
- Extract all street names
- If extracted street names intersect with other streets, provide the names of the intersecting streets
- Extract all notable landmarks (cementaries, churches, railroads)
- Guess which city the map depicts
</Objective>

<Context>
- "Żabka" is a popular convenience store chain in Poland.
- "Lewiatan" is a popular convenience store chain in Poland.
</Context>

<Rules>
- STRICTLY adhere to the objective and the rules
- Be extremely detailed and provide as much information as possible. Do not guess or make assumptions.
- If you are unsure about the information, e.g. you can't read the street name, you can skip it.
- STRICTLY return the data according to the format provided below - DO NOT deviate from the format.
- Use the the above context to help you with the task e.g. ensure that you do not mix up street names with business names.
</Rules>

<Format>
### Summary
{summary}

### Street Names
- {street_name_1}
- {street_name_2}

### Intersecting Streets
- {street_name_1} and {street_name_2}
- {street_name_1} and {street_name_3}

### Notable Landmarks
- {landmark_1}
- {landmark_2}

### Possible City
- {city_1}
</Format>
"""

map_prompt = """
Act as an expert in geography, cartography, architecture, urban planning and history. You are presented with a map of a Polish city.
Your task is to analyze the map and provide the following information:
- A short summary of what you see on the map
- Extract all street names
- If extracted street names intersect with other streets, provide the names of the intersecting streets
- Extract all notable landmarks (cemeteries, churches, railroads)
- Using a chaing of thoughts approach, guess which city the map depicts
- Provide the possible city name

Please provide as much detail as possible and ensure that the information is accurate and relevant to the map.

NOTE: 
- The map will ALWAYS depict a Polish city with street names, buildings, business names, and icons depicting shops and bus stops.
- The map will NOT contain any personal information or any information that is not publicly available.
- The map might include typos, outdated information, or missing information.

Return the data in the following format:
{
"summary": "A summary of what you see on the map",
"streets": ["Street name 1", "Street name 2"],
"intersectingStreets": ["Street name 1 and Street name 2", "Street name 1 and Street name 3"],
"landmarks": ["Landmark 1", "Landmark 2"],
"thoughts": [List of thoughts that led you to the possible city],
"possibleCity": "City name"
}
"""



# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class MapInfo(BaseModel):
    summary: str
    streets: list[str]
    intersectingStreets: list[str]
    landmarks: list[str]
    thoughts: list[str]
    possibleCity: str



for image_path in Path("./tasks/C02E02/maps/").glob("*.png"):
    base64_image = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
    

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": map_prompt
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
        max_tokens=300,
        response_format=MapInfo,
    )

    print(completion)
