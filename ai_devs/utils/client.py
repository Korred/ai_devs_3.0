from dataclasses import dataclass

import httpx


@dataclass
class AIDevsResponse:
    code: int
    message: str


class AIDevsClient:
    BASE_URL = "https://poligon.aidevs.pl/"
    VERIFY_DIR = "verify"

    def __init__(self, api_key: str, base_url: str = BASE_URL, verify_dir: str = VERIFY_DIR):
        self.api_key = api_key
        self.base_url = base_url
        self.verify_url = f"{base_url}{verify_dir}"

    def verify_task(self, task_id: str, data: str | dict | list) -> AIDevsResponse:
        payload = {"task": task_id, "apikey": self.api_key, "answer": data}

        response = httpx.post(self.verify_url, json=payload, timeout=120)

        if response.status_code != 200:
            raise Exception("Error while verifying task")

        return AIDevsResponse(**response.json())
