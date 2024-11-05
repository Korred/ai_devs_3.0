import httpx
from icecream import ic
from pyquery import PyQuery as pq
from openai import OpenAI

from dotenv import load_dotenv
import os

XYZ_URL = "https://xyz.ag3nts.org/"

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with httpx.Client() as client:
    response = client.get(XYZ_URL)

    if response.status_code != 200:
        raise Exception("Error while fetching data")

    # Create a pyquery object
    html = pq(response.text)

    # Extract the question)
    human_question = html("#human-question").text().split("\n")[1]

    ic(f"Question: {human_question}")

    system_prompt = """
    You are a helpful assistant that returns the corresponding year of a historical event.
    
    <rules>
    - Return only the year part of a date in the format YYYY e.g. 1912, 1945, 1990
    </rules>

    <examples>
    Q = Question, A = Answer

    - Q: When did the Titanic sink? A: 1912
    - Q: When did World War II end? A: 1945
    - Q: When did the Gulf War start? A: 1990
    </examples>
    """

    # Get the answer from OpenAI
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_question},
        ],
    )

    # Extract the answer
    answer = completion.choices[0].message.content

    ic(f"Answer: {answer}")

    form_data = {
        "username": "tester",
        "password": "574e112a",
        "answer": answer,
    }

    # Make a post request to the XYZ URL
    # Ensure to follow the redirects
    response = client.post(XYZ_URL, data=form_data, follow_redirects=True)

    ic(f"Response: {response.text}")
