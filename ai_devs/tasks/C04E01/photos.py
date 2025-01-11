import os
from enum import StrEnum
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from icecream import ic
from openai import OpenAI
from pydantic import BaseModel
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


class Action(StrEnum):
    ANALYZE = "ANALYZE"
    DARKEN = "DARKEN"
    BRIGHTEN = "BRIGHTEN"
    REPAIR = "REPAIR"
    BREAK = "BREAK"
    EXTRACT = "EXTRACT"
    SET_FINISHED = "SET_FINISHED"


class ImageTool(StrEnum):
    DARKEN = "DARKEN"
    BRIGHTEN = "BRIGHTEN"
    REPAIR = "REPAIR"


class ImageIssue(StrEnum):
    GLITCH = "GLITCH"
    TOO_DARK = "TOO_DARK"
    TOO_BRIGHT = "TOO_BRIGHT"


class ActionResponse(BaseModel):
    thoughts: str
    action: str
    params: list[str] | None


class CheckDescriptionResponse(BaseModel):
    valid: bool


class ExtractedURLs(BaseModel):
    urls: List[str]


class ImageDescriptionResponse(BaseModel):
    thoughts: str
    description: str
    issue: ImageIssue | None


class ImageState(BaseModel):
    url: str
    finished: bool = False
    actions_taken: List[Action] = []
    description: str = ""


class PersonDescription(BaseModel):
    thoughts: str
    description: str


class PhotoSearchAgent:
    def __init__(self):
        pass
        # self.image_collection = []

    def extract_image_urls(self, text: str) -> ExtractedURLs:
        system_prompt = """
            Extract all image URLs from the text.

            <CONTEXT>
            BASE_URL = "https://centrala.ag3nts.org/dane/barbara/"
            </CONTEXT>

            <RULES>
            - Extract all image URLs from the text e.g. https://example.com/image.jpg
            - Sometimes URLs might be split stating the BASE_URL and the image path (/image.jpg) separately. In such cases, combine the base-url and image path to form the complete URL.
            - If theBASE_URL is not provided, use the BASE_URL provided in the <CONTEXT> section above instead.
            - If the image URL is not a direct link to the image, skip it.
            - If the image URL is not a valid URL, skip it.
            - Return all extracted image URLs as stated in the <RETURN FORMAT> section.
            </RULES>

            <EXAMPLES>
            Input: "This is a sample text with an image testtest.jpg"
            Output: ["https://centrala.ag3nts.org/dane/barbara/testtest.jpg"]

            Input: "You can find the image in https://example.com/test/folder/a/ and the image is named image.png"
            Output: ["https://example.com/test/folder/a/image.png"]
            </EXAMPLES>

            <RETURN FORMAT>
            {
                "urls": [list of extracted image URLs]
            }
            </RETURN FORMAT>
        """

        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format=ExtractedURLs,
        )

        return response.choices[0].message.parsed

    def analyze_image(self, url: str) -> ImageDescriptionResponse:
        system_prompt = """
        Act as a image analysis expert and analyze the provided image.
        Describe the image in detail. The image might contain a glitch, be too dark, or too bright.
    
        <RULES>
        - The image analysis MUST be in Polish.
        - Analyze the provided image and describe it in detail.
        - If the image contains a glitch, is too dark, or too bright, mention the issue type.
        - Even if there are issues with the image and some parts are not visible, describe the ONLY the visible parts in detail.
          - When describing a person, be extra descriptive about their appearance:
            - describe their physical features such as hair color, eye color, body type, gender (male/female), etc.
            - describe the type and color of clothes they are wearing
            - describe if they have any accessories such as glasses, hats, etc.
            - notice any beauty marks, scars, or tattoos on their body
          - When describing the background, be extra descriptive about the location, lighting, objects, etc.
        - Follow the return format as stated in the <RETURN FORMAT> section
        </RULES>

        <RETURN FORMAT>
        {
            "thoughts": [Your general thoughts on the image],
            "description": [Your detailed description of the image],
            "issue": A string representing the issue with the image (if any) e.g. "GLITCH", "TOO_DARK", "TOO_BRIGHT" or null if there are no issues
        }
        </RETURN FORMAT>
        """

        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze the image below:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url,
                            },
                        },
                    ],
                },
            ],
            response_format=ImageDescriptionResponse,
        )

        return response.choices[0].message.parsed

    def modify_image(self, tool: ImageTool, url: str) -> str:
        if tool not in ImageTool:
            return "Invalid tool selected"

        filename = Path(url).name
        response = aidevs_client.query(
            "report", {"task": "photos", "answer": f"{tool} {filename}"}
        )

        return response.message

    def check_description(self, description: str) -> str:
        system_prompt = """
        Check if the provided image description contains the description of a woman.
        The description might be in Polish or English.
        
        <RULES>
        - Return either True or False based on whether the description contains the description of a woman
        - Do not return any other information
        - Follow the return format as stated in the <RETURN FORMAT> section
        </RULES>

        <RETURN FORMAT>
        {
            "valid": [boolean]
        }
        </RETURN FORMAT>
        """

        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": description},
            ],
            response_format=CheckDescriptionResponse,
        )

        valid = response.choices[0].message.parsed.valid

        message = f"""
        Description Check: {valid}
        The image description {"contains" if valid else "does not contain"} the description of a woman!
        """
        return message

    def extract_person_description(self, descriptions: List[str]) -> str:
        system_prompt = """
        Act as photo description expert. Given a list of image descriptions, extract the description of a woman.
        Some descriptions may be relevant, while others may not contain the description of woman. Retain only the relevant descriptions.

        <RULES>
        - Extract the description of a woman from the provided image descriptions.
        - Concentrate only on the person and extract all information about:
            - their physical features such as hair color, eye color, body type, etc.
            - the type and color of clothes they are wearing
            - if they have any accessories such as glasses, hats, etc.
            - beauty marks, scars, or tattoos on their body
        - Follow the return format as stated in the <RETURN FORMAT> section
        - Use "thoughts" to provide any general thoughts while extracting the final description.
        </RULES>

        <RETURN FORMAT>
        {
            "thoughts": Your general thoughts while extracting the description of the woman.
            "description": "Complete description of the person"
        }
        </RETURN FORMAT>
        """

        description_message = "\n------\n".join(descriptions)

        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": description_message},
            ],
            response_format=PersonDescription,
        )

        return response.choices[0].message.parsed.description

    def decide_next_action(
        self, messages: list[dict], current_state: str
    ) -> ActionResponse:
        current_state_message = "EMPTY"

        if current_state:
            current_state_message = "\n".join(
                [
                    f"URL: {key} - Finished? {value['finished']} - Actions taken: {value['actions_taken']}"
                    for key, value in current_state.items()
                ]
            )

        ic(current_state_message)

        system_prompt = f"""
        Act as a photo editing expert with the ability to analyze and decide the next action to take based on the provided user/system messages.
        Your goal is to ensure every image url is found (EXTRACT), analyzed (ANALYZE), and modified (DARKEN, BRIGHTEN, REPAIR) if needed, until all images are described (SET_FINISHED).
        If all images are finished, stop processing further (BREAK).

        <RULES>
        - Based on the provided <CURRENT STATE> and user messages, decide the next action to take.
        - **IMPORTANT**: Use only the actions provided in the <ACTIONS>, DO NOT PERFORM THEM, only decide the next action to take.
        - **IMPORTANT**: Each action can be performed only ONCE - check within the <CURRENT STATE> section if an action has already been performed on an image.
        - **IMPORTANT**: use BREAK only IF ALL images are done being analyzed (Finished=True).
        - **IMPORTANT**: Finish one image before moving to the next one.
        - The <CURRENT STATE> section will provide the current state of the images seen so far e.g. URLs, finished status, tools used
        - If a user message contains a free text in which an URL or file name is mentioned, use the EXTRACT action to extract the image URLs from the text.
        - SKIP finished images (Finished=True) and move to the next unfinished (Finished=False) image url in the <CURRENT STATE>.
        - If an image has an issue, use the appropriate tool to modify it (DARKEN, BRIGHTEN, REPAIR).
        - If a tool fails, try another tool or skip the image if all tools fail (SET_FINISHED).
        - If ALL images are finished (Finished=True), use the BREAK action to stop processing further.
        - Follow the return format as stated in the <RESPONSE FORMAT> section
        </RULES>

        <ACTIONS>
        - Action: EXTRACT
            Description: Used to extract image URLs from the text. Also use this action if only the file name is provided without the URL.
            Parameter: [entire user message]

        - Action: ANALYZE
            Description: Analyze the image and provide a detailed description (including any issues with the image).
            Condition: Use ANALYZE only if the image_url is in the <CURRENT STATE>.
            Parameter: [image_url]

        - Action: DARKEN
            Description: Darken / Decrease the brightness of an image that is too bright
            Parameter: [image_url]

        - Action: BRIGHTEN
            Description: Brigthen / Increase the brightness of an image that is too dark 
            Parameter: [image_url]

        - Action: REPAIR
            Description: Modify an image that has a glitch using the REPAIR tool
            Parameter: [image_url]
        
        - Action: BREAK
            Description: Stop processing further as all images are finished.
            Condition: USE ONLY IF ALL IMAGES/URL in the <CURRENT STATE> have finished set to True.
            Parameter: null

        - Action: SET_FINISHED
            Description: The current image is finished and we need to update the state in <CURRENT STATE> to skip it in the future.
            Condition: IGNORE this action if the image is already finished (Finished=True). Check <CURRENT STATE> before using this action.
            Parameter: [image_url]

        <RESPONSE FORMAT>
        {{
            "thoughts": [Your general thoughts on the messages],
            "action": [Action to take - EXTRACT, ANALYZE, DARKEN, BRIGHTEN, REPAIR, BREAK, SET_FINISHED],
            "params": [List of parameters for the action] or null
        }}
        </RESPONSE FORMAT>

        <EXAMPLES>
        Input: 
            Current State: EMPTY,
            User Message: "Here are some images: https://example.com/image1.jpg, https://example.com/image/2.jpg"
        Output: {{
            "thoughts": "Extracting image URLs from the text",
            "action": "EXTRACT",
            "params": ["Here are some images: https://example.com/image1.jpg, https://example.com/image/2.jpg"]
        }}
        ---------
        Input:
            Current State: 
                URL: https://example.com/image1.jpg - Finished? True - Actions taken: ["ANALYZE", "DARKEN", "BRIGHTEN", "REPAIR"]
                URL: https://example.com/image2.jpg - Finished? False - Actions taken: ["ANALYZE", "DARKEN"]
            User Message: "
                Action performed: SET_FINISHED
                
                Image https://example.com/image1.jpg has been finished.

                Ensure to skip this image in the future.
                Check for any other images that need processing.
            "

        Output: {{
            "thoughts": "I have finished processing the image https://example.com/image1.jpg. However, there are more unfinished images available. https://example.com/image2.jpg was previously ANALYZED and DARKENED. There is no need to process the image further. I need to update the image state to skip it in the future.",
            "action": "SET_FINISHED",
            "params": ["https://example.com/image2.jpg"]
        }}
        -----------
        Input:
            Current State: URL: https://example.com/image1.jpg - Finished? True - Actions taken: ["ANALYZE", "DARKEN", "LIGHTEN", "REPAIR"]
            User Message: "
                Action performed: DARKEN
                Tool response: Your tool didn't work on the image. Try something else.
                
                The tool response may contain an updated image URL or a message indicating the tool did not work.
                Use EXTRACT action to extract the new image URL or file name is available.
                "
            
        Output: {{
            "thoughts": "The image tool didn't work. I have already ANALYZED the image and tried DARKEN, LIGHTEN, and REPAIR tools, and based on the current state, the image is finished. There are no more images to process. I will stop processing further.",
            "action": "BREAK",
            "params": null
        }}
        -----------
        Input:
            Current State: URL: https://example.com/image1.jpg - Finished? False - Actions taken: []
            User Message: "
                Action performed: EXTRACT

                Extracted Image URLs: [https://example.com/image1.jpg]
            "

        Output: {{
            "thoughts": "I have found the following URLs: https://example.com/image1.jpg. I will now analyze the image.",
            "action": "ANALYZE",
            "params": ["https://example.com/image1.jpg"]
        }}
        ----------
        Input:
            Current State: URL: https://example.com/image1.jpg - Finished? False - Actions taken: ["ANALYZE"]
            User Message: "
                Action performed: ANALYZE
                Issue: "TOO_BRIGHT"
                Description of the image: "It seems there is a person on the left side of the image but the image is too bright to see any details."
            "

        Output: {{
            "thoughts": "The image is too bright. I will now use the DARKEN tool to modify the image.",
            "action": "DARKEN",
            "params": ["https://example.com/image1.jpg"]
        }}
        ---------
        Input:
            Current State: URL: https://example.com/image1.jpg - Finished? False - Actions taken: ["ANALYZE", "BRIGHTEN"]
            User Message: "
                Action performed: BRIGHTEN
                Tool response: Yo! Here is the modified image: https://example.com/image1_brightened.jpg

                The tool response may contain an updated image URL or a message indicating the tool did not work.
                Use EXTRACT action to extract the new image URL or file name is available.
            "

        Output: {{
            "thoughts": "The image has been modified using the BRIGHTEN tool. There is a new image available for further analysis. I need to extract the image URL from the text.",
            "action": "EXTRACT",
            "params": ["Yo! Here is the modified image: https://example.com/image1_brightened.jpg"]
        }}
        ---------
        Input:
            Current State: 
                URL: https://example.com/image1.jpg - Finished? False - Actions taken: ["ANALYZE", "DARKEN", "BRIGHTEN", "REPAIR"]
                URL: https://example.com/image2.jpg - Finished? False - Actions taken: []
            User Message: "
                Action performed: REPAIR
                Tool response: "The image looks worse than before. Try something else."

                The tool response may contain an updated image URL or a message indicating the tool did not work.
                Use EXTRACT action to extract the new image URL or file name is available.
            "

        Output: {{
            "thoughts": "The image tool didn't work. I have already ANALYZED the image and tried DARKEN, LIGHTEN, and REPAIR tools, and based on the current state, the image is finished. I need to update the image state to skip it in the future.",
            "action": "SET_FINISHED",
            "params": ["https://example.com/image1.jpg"]
        }}

        </EXAMPLES>

        <CURRENT STATE>
        {current_state}
        </CURRENT STATE>
        """

        active_messages = [
            {"role": "system", "content": system_prompt},
        ]

        active_messages.extend(messages)

        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=active_messages,
            response_format=ActionResponse,
            temperature=0.7,
        )

        return response.choices[0].message.parsed

    def process(self, text: str) -> str:
        current_state = {}
        messages = [
            {"role": "user", "content": text},
        ]

        ic("Starting the process...")

        # Just pass the last three messages to the agent
        # The agent will decide the next action based on the message context and the current state

        while True:
            response = self.decide_next_action(messages[-3:], current_state)

            ic(response)

            action = response.action
            params = response.params

            messages.append({"role": "system", "content": response.thoughts})

            message = ""

            if action == Action.EXTRACT:
                extract = self.extract_image_urls(response.params[0])

                message = f"""
                Action performed: EXTRACT

                Extracted Image URLs:
                {extract.urls}
                """

                messages.append({"role": "user", "content": message})

                for url in extract.urls:
                    current_state[url] = {
                        "url": url,
                        "finished": False,
                        "actions_taken": [],
                        "description": "",
                    }

            elif action == Action.ANALYZE:
                analysis = self.analyze_image(params[0])

                message = f"""
                Action performed: ANALYZE

                Issue:{analysis.issue if analysis.issue else "No issue detected"}
                Description of the image: 
                {analysis.description}
                """

                messages.append({"role": "user", "content": message})

                current_state[params[0]]["actions_taken"].append(action)
                current_state[params[0]]["description"] = analysis.description

            elif action in (Action.DARKEN, Action.BRIGHTEN, Action.REPAIR):
                tool_message = self.modify_image(action, params[0])

                message = f"""
                Action performed: {action}

                Tool response: {tool_message}

                The tool response may contain an updated image URL or a message indicating the tool did not work.
                Use EXTRACT action to extract the new image URL or file name is available.
                """

                messages.append({"role": "user", "content": message})

                current_state[params[0]]["actions_taken"].append(action)

            elif action == Action.SET_FINISHED:
                message = f"""
                Action performed: SET_FINISHED

                Image {params[0]} has been finished.

                Ensure to skip this image in the future.
                Check for any other images that need processing.
                """

                messages.append({"role": "user", "content": message})

                current_state[params[0]]["finished"] = True

            elif action == Action.BREAK:
                break

            ic(message)

        descriptions = [value["description"] for value in current_state.values()]
        person_description = self.extract_person_description(descriptions)
        return person_description


if __name__ == "__main__":
    # Get start text
    response = aidevs_client.query("report", {"task": "photos", "answer": "START"})
    ic(response)

    # Start analyzing the text
    agent = PhotoSearchAgent()
    response = agent.process(response.message)

    ic(response)

    # Save response to file
    with open("person_description.txt", "w", encoding="utf-8") as f:
        f.write(response)

    ic("Process completed successfully!")

    # Send the response back to the server
    response = aidevs_client.query("report", {"task": "photos", "answer": response})

    ic(response)
