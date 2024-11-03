from dataclasses import dataclass

import httpx


@dataclass
class AIDevsResponse:
    code: int
    message: str


class AIDevsClient:
    BASE_URL = "https://poligon.aidevs.pl/"
    VERIFY_URL = BASE_URL + "verify"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def verify_task(self, task_id: str, data: str | dict | list) -> AIDevsResponse:
        payload = {"task": task_id, "apikey": self.api_key, "answer": data}

        response = httpx.post(self.VERIFY_URL, json=payload, timeout=120)

        if response.status_code != 200:
            raise Exception("Error while verifying task")

        return AIDevsResponse(**response.json())
