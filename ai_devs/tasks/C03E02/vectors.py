import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from icecream import ic
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from utils.client import AIDevsClient

COLLECTION_NAME = "weapons_collection"
EMBEDDING_DIM = 3072
DOCUMENTS_PATH = Path("./tasks/C03E02/documents/")

DOCUMENT_TEMPLATE = """
----
Data: {publication_date}
Broń: {weapon_name}
Wzmianka: {text}
----
"""


class Response(BaseModel):
    value: str


# Load environment variables from .env file
load_dotenv()


# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qd_client = QdrantClient(host="localhost", port=6333)
aidevs_client = AIDevsClient(
    api_key=os.getenv("AIDEVS_API_KEY"),
    base_url="https://centrala.ag3nts.org/",
    verify_dir="report",
)


# Create a collection
def create_collection(collection_name: str, client: QdrantClient):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM, distance=Distance.COSINE, on_disk=True
            ),
        )


def get_weapon_name(message: str) -> str:
    system_prompt = """
    Your task is to extract the weapon name from the given text. 

    <RULES>
    1. The weapon name is always mentioned in the text.
    2. The text is in Polish - do not translate it.
    3. Return the weapon name as it is mentioned in the text, without any additional information.
    4. The response should follow the format in the <FORMAT> section.
    </RULES>

    <FORMAT>
    {
        "value": "Weapon Name"
    }
    </FORMAT>

    <EXAMPLE>
    Input: "Podczas II wojny światowej żołnierze rosyjscy używali strzelby Mosin-Nagant."
    Output: 
    {
        "value": "Mosin-Nagant"
    }
    </EXAMPLE>
    """

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        response_format=Response,
    )

    return completion.choices[0].message.parsed


def chunk_text_into_paragraphs(text: str) -> list:
    return text.split("\n\n")


def answer_question(question: str, relevant_documents: str):
    system_prompt = f"""
    Your task is to answer a question based on the given documents.

    <DOCUMENTS>
    {relevant_documents}
    </DOCUMENTS>

    <RULES>
    1. The answer should be based on the information provided in the documents.
    2. The answer should be a direct response to the question.
    3. Response should be in Polish.
    4. Your reply should be concise and to the point e.g. when asked for a date, provide a date.
    5. The response should follow the format in the <FORMAT> section.
    6. Return dates in the format "YYYY-MM-DD".
    </RULES>

    <FORMAT>
    {{
        "value": "Answer"
    }}
    </FORMAT>
    """

    ic(system_prompt)

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        response_format=Response,
    )

    return completion.choices[0].message.parsed


if __name__ == "__main__":
    create_collection(COLLECTION_NAME, qd_client)

    collection = qd_client.get_collection(COLLECTION_NAME)

    if collection.points_count == 0:
        points = []

        # Load documents from the documents folder
        for file in DOCUMENTS_PATH.glob("*.txt"):
            ic(f"Processing file: {file.name}")

            document_id = str(uuid.uuid4())
            publication_date = file.stem
            text = file.read_text(encoding="utf-8")

            # Chunk the text into paragraphs
            paragraphs = chunk_text_into_paragraphs(text)

            # First paragraph contains the weapon name
            ic("Getting weapon name")
            weapon_name = get_weapon_name(paragraphs[0]).value

            for e, paragraph in enumerate(paragraphs):
                ic(f"Processing paragraph {e}")
                vector = (
                    openai_client.embeddings.create(
                        input=paragraph, model="text-embedding-3-large"
                    )
                    .data[0]
                    .embedding
                )

                ic("Embedding created")

                ic("Creating point")
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "paragraph_index": e,
                        "weapon_name": weapon_name,
                        "text": paragraph,
                        "publication_date": publication_date,
                    },
                )

                points.append(point)

        ic("Upserting points")
        qd_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

    ic("Collection created and points upserted")

    question = (
        "W raporcie, z którego dnia znajduje się wzmianka o kradzieży prototypu broni?"
    )

    ic("Getting question embedding")

    question_embedding = (
        openai_client.embeddings.create(input=question, model="text-embedding-3-large")
        .data[0]
        .embedding
    )

    hits = qd_client.search(
        collection_name=COLLECTION_NAME, query_vector=question_embedding, limit=3
    )

    # Prepare documents for the question
    relevant_documents = [
        DOCUMENT_TEMPLATE.format(
            publication_date=hit.payload["publication_date"],
            weapon_name=hit.payload["weapon_name"],
            text=hit.payload["text"],
        )
        for hit in hits
    ]
    relevant_documents = "\n".join(relevant_documents)

    answer = answer_question(question, relevant_documents).value

    ic(answer)

    # Submit answer to the AIDevs API
    response = aidevs_client.verify_task("wektory", answer)
    ic(response)
