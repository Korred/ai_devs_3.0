from dataclasses import dataclass

import httpx


@dataclass
class AIDevsResponse:
    code: int
    message: str

@dataclass
class AIDevsDBResponse:
    reply: list[dict] | None
    error: str

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
            print(response.text)
            raise Exception("Error while verifying task")

        return AIDevsResponse(**response.json())
    
    def query_db(self, query: str, task: str) -> AIDevsResponse:
        payload = {
            "task": task,
            "apikey": self.api_key,
            "query": query
        }

        response = httpx.post(f"{self.base_url}apidb", json=payload, timeout=120)

        if response.status_code != 200:
            print(response.text)
            raise Exception("Error while querying database")

        return AIDevsDBResponse(**response.json())

    def query(self, endpoint: str, payload: dict) -> AIDevsResponse:
        payload["apikey"] = self.api_key

        response = httpx.post(f"{self.base_url}{endpoint}", json=payload, timeout=120)

        if response.status_code != 200:
            print(response.text)
            raise Exception("Error while querying database")
        
        return AIDevsResponse(**response.json())