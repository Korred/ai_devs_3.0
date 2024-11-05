import os

import httpx
from dotenv import load_dotenv
from icecream import ic
from utils.client import AIDevsClient

# Load environment variables from .env file
load_dotenv()

# Get api key from environment variables
AIDEVS_API_KEY = os.getenv("AIDEVS_API_KEY")

client = AIDevsClient(AIDEVS_API_KEY)

# Get poligon task data
task_id = "POLIGON"
task_data_url = "https://poligon.aidevs.pl/dane.txt"

data = httpx.get(task_data_url).text
data_as_list = data.strip().split("\n")
ic(data_as_list)

# Verify task
response = client.verify_task(task_id, data_as_list)
ic(response)
