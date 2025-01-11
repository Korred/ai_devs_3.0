import os
from typing import List


import httpx
from dotenv import load_dotenv
from openai import OpenAI
from utils.client import AIDevsClient
from icecream import ic

from pydantic import BaseModel






class RobotVisualElements(BaseModel):
    shape: list[str]
    size: list[str]
    keyFeatures: list[str]
    material_texture: list[str]
    function_movement: list[str]
    distinctive_elements: list[str]
    location: list[str]

def extract_robot_visual_elements(description: str) -> RobotVisualElements:

    visual_elements_extract = """
    Act as a keen observer with a photographer's eye. 
    Based on the user-provided description, extract only the essential visual information about the robot's physical appearance and location. 

    Focus specifically on the following visual elements:

    Shape: Identify the general shape or form of the robot (e.g., spherical, humanoid, rectangular).
    Size: Note any indications of size if mentioned (e.g., small, large, human-sized).
    Key Features: Extract details about significant components or features, such as visible sensors, cameras, limbs, or other appendages.
    Material and Texture: Note any references to the materials or surface characteristics (e.g., metal, plastic, smooth, rough).
    Function or Movement: If there's any indication of the robot's movement or posture, summarize it (e.g., floating, walking).
    Distinctive Elements: Any unique or unusual markings, colors, or symbols that might help identify this robot visually.
    Location: If the description includes a specific location or environment, mention it.

    <Rules>
    - Ignore any non-visual commentary or subjective opinions unrelated to appearance (such as emotions, humor, or personal anecdotes).
    - Focus only on gathering the visual data elements needed to create an image of the robot.
    - Ensure your response is concise and STRICTLY adheres to the output format below.
    </Rules>

    <Format>
    {
        "shape": [list of shapes],
        "size": [list of sizes],
        "keyFeatures": [list of features],
        "material_texture": [list of materials or textures],
        "function_movement": [list of functions or movements],
        "distinctive_elements": [list of distinctive elements e.g., colors, markings],
        "location": [list of locations, environments, scenarios, distinct elements of the setting]
    }
    </Format>

    <Example>
    {
        "shape": ["spherical"],
        "size": ["small"],
        "keyFeatures": ["multiple cameras", "sensors"],
        "material_texture": ["metal", "abs plastic", "rubber"],
        "function_movement": ["hovering"],
        "distinctive_elements": ["red color", "glowing eyes"]
        "location": ["research facility", "apocalypse scenario", "destroyed generator"]
    }
    </Example>
    """

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": visual_elements_extract
            },
            {
                "role": "user",
                "content": description
            }
        ],
        response_format=RobotVisualElements,
    )

    return completion.choices[0].message.parsed


def create_dall_e_prompt(robot_visual_elements: RobotVisualElements, tone: List[str] = [], color_scheme: List[str] = [], lighting_atmosphere: List[str] = []) -> str:
    system_prompt = f"""
    Act as a visual artist and expert creating specialized DALL-E prompts for generating images of robots.
    Prioritize accuracy in representing the visual elements provided, but adapt to include stylistic choices that align with the described characteristics. 
    For any missing details, use a logical interpretation that aligns with the general theme.

    Shape: {", ".join(robot_visual_elements.shape)}
    Size: {", ".join(robot_visual_elements.size)}
    Key Features: {", ".join(robot_visual_elements.keyFeatures)}
    Material and Texture: {", ".join(robot_visual_elements.material_texture)}
    Function or Movement: {", ".join(robot_visual_elements.function_movement)}
    Distinctive Elements: {", ".join(robot_visual_elements.distinctive_elements)}
    Location: {", ".join(robot_visual_elements.location)}

    Additionally, incorporate any of the following stylistic elements for tone and color scheme:

    Tone: {", ".join(tone)}
    Color Scheme: {", ".join(color_scheme)}
    Lighting and Atmosphere: {", ".join(lighting_atmosphere)}

    Use your discretion to balance accuracy with visual appeal, highlighting the robot’s distinct and futuristic qualities.

    <Rules>
    - ONLY return the generated prompt - NOTHING ELSE.
    - Focus on creating a detailed and visually engaging prompt that reflects the robot's characteristics.
    - Incorporate stylistic choices that enhance the visual representation without distorting the core elements.
    - Ensure the prompt is clear, concise, and aligns with the provided visual data.
    - Avoid introducing unrelated or contradictory elements that deviate from the original description.
    </Rules>
    """

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
    )

    return completion.choices[0].message.content


def generate_robot_image(prompt: str) -> str:

    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
        )

    image_url = response.data[0].url

    return image_url


# Load environment variables from .env file
load_dotenv()

# Initialize clients
aidevs_client = AIDevsClient(
    api_key=os.getenv("AIDEVS_API_KEY"),
    base_url="https://centrala.ag3nts.org/",
    verify_dir="report",
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
AIDEVS_API_KEY = os.getenv("AIDEVS_API_KEY")
FILE_URL = f"https://centrala.ag3nts.org/data/{AIDEVS_API_KEY}/robotid.json"

with httpx.Client() as client:
    robotid_json = client.get(FILE_URL).json()

description = robotid_json["description"]


ic(description)

elements = extract_robot_visual_elements(description)
ic(elements)

# prompt = create_dall_e_prompt(elements, tone=["fallout", "dystopian"], color_scheme=["desert", "rust"], lighting_atmosphere=["dim", "overcast"])
prompt = create_dall_e_prompt(elements)
ic(prompt)

image_url = generate_robot_image(prompt)
ic(image_url)

# Submit the generated image to the AIDevs API
response = aidevs_client.verify_task("robotid", image_url)
ic(response)