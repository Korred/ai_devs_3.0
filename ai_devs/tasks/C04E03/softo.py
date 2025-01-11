import hashlib
import os
import uuid

from dotenv import load_dotenv
from icecream import ic
from openai import OpenAI
from utils.client import AIDevsClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Record, Filter, MatchValue, FieldCondition

from pydantic import BaseModel

import httpx


# Load environment variables from .env file
load_dotenv()

# Initialize various clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=1*60, max_retries=3)
aidevs_client = AIDevsClient(
    api_key=os.getenv("AIDEVS_API_KEY"),
    base_url="https://centrala.ag3nts.org/",
    verify_dir="report",
)
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "websites"
EMBEDDING_DIM = 3072
QUESTIONS_URL = (
    f"https://centrala.ag3nts.org/data/{os.getenv("AIDEVS_API_KEY")}/softo.json"
)


class ParseParams(BaseModel):
    url: str


class ParseResponse(BaseModel):
    parsed_url: str
    markdown: str
    new_urls: list[str]


class QueryParams(BaseModel):
    url: str


class QueryResponse(BaseModel):
    queries: list[str]
    texts: list[str]


class AnswerParams(BaseModel):
    url: str
    text: str
    document_ids: list[str]


class AnswerResponse(BaseModel):
    thoughts: str
    found: bool
    answer: str


class ActionResponse(BaseModel):
    thoughts: str
    action: str
    parse: ParseParams | None
    query: QueryParams | None
    answer: AnswerParams | None


class State(BaseModel):
    unparsed_urls: set[str]
    parsed_urls: set[str] = set()
    checked_urls: set[str] = set()
    question: str = ""


# Create a collection
def create_collection(collection_name: str, client: QdrantClient):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM, distance=Distance.COSINE, on_disk=True
            ),
        )


class WebsiteQuestionAgent:
    MAX_ACTIONS = 20

    def __init__(self, url: str):
        self.state = State(unparsed_urls=[url])

    @staticmethod
    def hash_url(url):
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def add_document_to_collection(self, collection_name: str, document: ParseResponse) -> list[Record]:
        ic("Creating embedding...")
        vector = (
            openai_client.embeddings.create(
                input=document.markdown, model="text-embedding-3-large"
            )
            .data[0]
            .embedding
        )

        point_id = str(uuid.uuid4())
        document_id = self.hash_url(document.parsed_url)

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "url": document.parsed_url,
                "document_id": document_id,
                "markdown": document.markdown,
                "related_urls": document.new_urls,
            },
        )

        ic("Adding document to collection...")
        qdrant_client.upsert(collection_name=collection_name, points=[point], wait=True)

        ic("Retrieving document from collection...")
        return qdrant_client.retrieve(collection_name, [point_id])

    @staticmethod
    def get_documents_by_url(collection_name:str, url: str) -> list[Record]:
        documents = []
        has_more = True
        offset = None

        while has_more:
            scroll_result, next_page_offset = qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,  # Adjust the limit as needed
                offset=offset,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="url",
                            match=MatchValue(value=url)
                        )
                    ]
                ),
                with_payload=True
            )
            documents.extend(scroll_result)
            offset = next_page_offset
            has_more = offset is not None

        return documents

    def parse_url(self, url: str) -> ParseResponse:
        ic("Parsing URL with Firecrawl...")

        timeout = httpx.Timeout(10.0, read=None)

        response = httpx.post(
            url="https://api.firecrawl.dev/v1/scrape",
            headers={
                "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "url": url,
                "formats": ["markdown", "links"],
                "onlyMainContent": False,
            },
            timeout=timeout,
        ).json()

        return ParseResponse(
            parsed_url=url,
            markdown=response['data']['markdown'],
            new_urls=response['data']['links'],
        )
    
    def get_answer(self, question: str, documents: list[Record]) -> AnswerResponse:
        system_prompt = """
        Your task is to answer the user provided question based on the given context.

        <RULES>
        1. The answer should be based on the context provided e.g. use ONLY the information from the context.
        2. The answer should be relevant to the question.
        3. Answer in the same language as the question e.g. if the question is in English, answer in English.
        4. The answer should be short and precise. Stick to the main point.
        5. The response should follow the format in the <FORMAT> section
        6. Use "thoughts" to provide any general thoughts while extracting the answer from the context. They should be in English.
        </RULES>

        <FORMAT>
        {{
            "thoughts": "Any thoughts or reasoning behind the answer",
            "found": (bool) Whether the answer was found in the context,
            "answer": (str) Short/pricise answer to the question or an empty string if the answer was not found.
        }}
        </FORMAT>

        <EXAMPLE>
        Input:
            Question: "Ile lat ma Jan Kowalski?"
            Context: 
            "Jan Kowalski urodzil sie w Warszawie"
            -----
            "Jan Kowalski ma 30 lat i jest inzynierem"

        Output:
        {{
            "thoughts": "Jan Kowalski's age, place of birth and profession are mentioned in the context. He is 30 years old.",
            "found": true,
            "answer": "Jan Kowalski ma 30 lat"
        }}
        ----
        Input:
            Question: "Kto jest prezydentem Polski?"
            Context: 
            "Ala ma kota"
            -----
            "Kot jest czarny"

        Output:
        {{
            "thoughts": "The context does not mention anything about the president of Poland.",
            "found": false,
            "answer": ""
        }}
        </EXAMPLE>
        """

        user_message = f"""
        Question: "{question}"
        Context: 
        {'\n-----\n'.join([f"{document.payload['markdown']}" for document in documents])}
        """

        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=AnswerResponse,
        )

        return response.choices[0].message.parsed

    def decide_action(self, messages: list[dict]) -> ActionResponse:

        system_prompt = f"""
        Act as an AI assistant with the ability to analyze the current conversation context and decide the next action to be taken.
        Your ultimate goal is to answer the user's question based on the provided context (content of parsed URLs).

        <RULES>
        - Based on the provided <CURRENT_STATE> and user messages, decide the next action to take in order to answer the user's question.
        - **IMPORTANT**: Use only the actions provided in the <ACTIONS> section and DO NOT PERFORM THEM! Only decide the next action to take.
        - The <CURRENT_STATE> will store the user's question, URLs of parsed/unparsed websites that might contain the necessary information to generate an answer and URLs that have already been checked and found to be irrelevant to the user's question.
        - DO NOT consider URLs that seem irrelevant to the user's question or that seem sketchy e.g. bot detection pages, error pages, etc.
        - DO NOT consider URLs that were already checked and found to be irrelevant to the user's question.
        - If it seems like there is not enough context to answer the user's question, take the MOST LIKELY unparsed URL and parse it to get more context.
        - Follow the return format as stated in the <RESPONSE_FORMAT> section
        </RULES>

        <ACTIONS>
        - Action: parse
            Description: Parse any unparsed URL and add it to the collection so that it can be queried later. Only use this action if the URL seems relevant to the user's question.
            Parameters: {{url: "unparsed URL that needs to be parsed"}}

        - Action: query
            Description: Query the collection for relevant documents based on the provided URL.
            Parameters: {{url: "parsed URL to be queried"}}

        - Action: answer
            Description: Answer the user's question based on the context provided in the parsed URLs.
            Parameters: {{url: "parsed URL that was used for the relevant 'query' action", text: "User's question", document_ids: [List of relevant document chunk ids that were returned by the 'query' action]}}
        </ACTIONS>

        <RESPONSE_FORMAT>
        {{
            "thoughts": "Any thoughts or reasoning behind the action to be taken",
            "action": "Action to be taken e.g. one of 'parse', 'query', 'answer'",
            "parse": (Optional) Parameters for the parse action - only if the action is 'parse',
            "query": (Optional) Parameters for the query action - only if the action is 'query',
            "answer": (Optional) Parameters for the answer action - only if the action is 'answer'
        }}
        </RESPONSE_FORMAT>

        <CURRENT_STATE>
            Question: {self.state.question}
            Parsed URLs: {self.state.parsed_urls}
            Unparsed URLs: {self.state.unparsed_urls}
            Checked URLs: {self.state.checked_urls}
        </CURRENT_STATE>
        """

        active_messages = [
            {"role": "system", "content": system_prompt},
        ]

        active_messages.extend(messages)
  
        ic("Deciding action...")
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=active_messages,
            response_format=ActionResponse,
            temperature=0.7,
            
        )

        return response.choices[0].message.parsed

    def process(self, question: str):
        # Add question to the state
        self.state.question = question

        messages = [
            {"role": "user", "content": question},
        ]

        ic("Starting process...")

        actions_taken = 0
        while actions_taken < self.MAX_ACTIONS:
            ic("Actions taken:", actions_taken)
            ic("Current state:", self.state)
            print()
            response = self.decide_action(messages[-3:])
            ic(response)

            messages.append({"role": "system", "content": response.thoughts})

            action = response.action

            user_message_template = """
            Action taken: {action}
            Response:
               {response}
            """

            if action == "parse":
                ic("Parsing...")
                # Before parsing, check if the URL has already been parsed
                if records := self.get_documents_by_url(COLLECTION_NAME, response.parse.url):
                    message = f"The URL {response.parse.url} has been successfully parsed. It is now available for querying."
                else:
                    # Use Firecrawl to parse the URL and get all the relevant information
                    parsed_response = self.parse_url(response.parse.url)

                    # Add the parsed document to the collection
                    records = self.add_document_to_collection(COLLECTION_NAME, parsed_response)

                    message = f"The URL {response.parse.url} has been successfully parsed. It is now available for querying."

                # Add the parsed URL to the list of already parsed URLs
                self.state.parsed_urls.add(response.parse.url)

                # Add URLs that were found in the already parsed URL to the unparsed URLs
                # ? Even if a document has been split into multiple chunks, all of the chunks would have the same related_urls in the payload/metadata
                self.state.unparsed_urls.update(
                    records[0].payload.get("related_urls", [])
                )

                # Remove already parsed URL from the unparsed URLs e.g. ensure that a URL is not parsed multiple times
                self.state.unparsed_urls -= self.state.parsed_urls

                messages.append(
                    {
                        "role": "system",
                        "content": user_message_template.format(
                            action=action, response=message
                        ),
                    }
                )

            elif action == "query":
                ic("Querying...")
                records = self.get_documents_by_url(COLLECTION_NAME, response.query.url)

                if not records:
                    message = f"No relevant data found for the URL {response.query.url}. Please parse the URL first."
                else:
                    message = f"""
                    Found {len(records)} document chunks for the URL {response.query.url}.
                    Pass the following chunk IDs as context when trying to answer: {", ".join([record.id for record in records])}
                    """

                messages.append(
                    {
                        "role": "system",
                        "content": user_message_template.format(
                            action=action, response=message
                        ),
                    }
                )

            elif action == "answer":
                ic("Answering...")
                
                records = qdrant_client.retrieve(COLLECTION_NAME, response.answer.document_ids)

                answer = self.get_answer(question, records)

                if answer.found:
                    return answer.answer

                message = f"The answer to the question was not found in the provided context. Thoughts: {answer.thoughts}"
                messages.append(
                    {
                        "role": "system",
                        "content": user_message_template.format(
                            action=action, response=message
                        ),
                    }
                )

                self.state.checked_urls.add(response.answer.url)
                self.state.parsed_urls.discard(response.answer.url)

            # Show the last two messages (system + action response)
            ic(messages[-1])

            actions_taken += 1

        return "I'm sorry, I couldn't find an answer to your question."


if __name__ == "__main__":
    # First create and get a new collection if it doesn't exist
    create_collection(COLLECTION_NAME, qdrant_client)
    collection = qdrant_client.get_collection(COLLECTION_NAME)

    questions = httpx.get(QUESTIONS_URL).json()

    ic(questions)
    answers = {}
    for qid, question in questions.items():
        agent = WebsiteQuestionAgent("https://softo.ag3nts.org/")
        answers[qid] = agent.process(question)

    ic(answers)

    # Send the response back to the server
    response = aidevs_client.query("report", {"task": "softo", "answer": answers})


    ic(response)

