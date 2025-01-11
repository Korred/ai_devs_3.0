import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from utils.client import AIDevsClient


class FactKeywords(BaseModel):
    name: str
    keywords: list[str]

class ContainsFacts(BaseModel):
    containsFacts: bool

class ReportKeywords(BaseModel):
    keywords: list[str]


FACTS_PATH = Path("./tasks/C03E01/facts/")
REPORTS_PATH = Path("./tasks/C03E01/reports/")


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




def contains_facts(text: str) -> bool:
    system_prompt = """
    You are tasked with analyzing a document to determine if it contains any facts about specific individuals.
    Return "true" if the document contains any facts about individuals, and "false" otherwise.

    <EXAMPLE>
    Input:
    Nothing to see here.
    
    Output:
    {
        "containsFacts": false
    }

    Input:
    John Doe is a software engineer at a tech company.

    Output:
    {
        "containsFacts": true
    }

    </EXAMPLE>

    <FORMAT>
    {
        "containsFacts": boolean
    }
    </FORMAT>
    """

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
        response_format=ContainsFacts
    )

    fact_check = completion.choices[0].message.parsed

    return fact_check.containsFacts

def extract_name_keywords_from_fact(fact: str) -> FactKeywords:


    system_prompt = """
    You are tasked with analyzing a document about a specific individual.

    <TASK>
    - Extract the person's full name.
    - Generate a list of relevant keywords (in Polish, nominative case) that describe key details about the person, their actions, roles, and relationships.
    </TASK>

    <STEPS>
    1. Identify the person's full name.
    2. Extract concise keywords (one or two words) directly taken from the text, including:
        - The person's roles, professions, or attributes.
        - Significant events (dates=YYYY-MM-DD, time=HH:MM), organizations, technologies, or locations related to the person (especially proper nouns).
        - Full names of other notable individuals mentioned.
    3. Make sure all keywords are in the nominative case (the base form of the word) and are relevant to the context.
    4. Deduplicate the extracted keywords.
    5. Check the spelling and correctness of the keywords, ensuring they exactly match the words used in the text (besides the nominative case).
    6. Format the result as shown in the <FORMAT> section.
    </STEPS>

    <FORMAT>
    {
    "name": "Person Name",
    "keywords": ["keyword1", "keyword2", "keyword3"]
    }
    </FORMAT>

    <EXAMPLE>
    Input:

    Anna Nowak, doświadczona oficer bezpieczeństwa w fabryce chemicznej w Krakowie, przeprowadziła ewakuację podczas wycieku chemikaliów.
    Współpracowała ściśle z Janem Kowalskim, kierownikiem fabryki, aby skoordynować procedury bezpieczeństwa.
    Anna jest znana ze swojej wiedzy eksperckiej w zakresie reagowania kryzysowego i bezpieczeństwa chemicznego.
    ----
    Output:

    {
    "name": "Anna Nowak",
    "keywords": ["Anna Nowak", "ewakuacja", "fabryka chemiczna", "Jan Kowalski", "Kraków", "oficer", "wyciek chemikaliów"]
    }
    </EXAMPLE>
    """



    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Fact:\n{fact}"},
        ],
        temperature=0.2,
        response_format=FactKeywords
    )

    return completion.choices[0].message.parsed

def extract_keywords_from_report(filename: str, report: str, facts: str) -> ReportKeywords:
    system_prompt_template = """
    You are an advanced text analysis assistant. 
    Your role is to analyze reports, extract relevant keywords (in Polish), and enhance them using a predefined list of facts. 
    Use the facts provided below as part of your knowledge base for enhancing keyword generation:

    <FACTS>
    {facts}
    </FACTS>

    <TASK>
    - Analyze the provided report (including the title).
    - Extract relevant keywords (in Polish, nominative case) that describe key details about the report, directly taken from the text.
    - Enhance the keywords using the provided facts.
    </TASK>

    <STEPS>
    1. Carefully read and understand the content of the provided report (including the title).
    2. Extract an initial list of concise keywords (one or two words) directly taken from the text, including:
        - The main subjects, topics, or themes discussed in the report.
        - Significant events (dates as YYYY-MM-DD, times as HH:MM), organizations, technologies, or locations mentioned in the report (especially proper nouns).
        - Full names of individuals mentioned.
        - Report number and sector as keywords.
    3. Ensure all keywords are in the nominative case and are relevant to the context.
    4. Enhance Keywords Using Facts:
        - Check if any person from the <FACTS> list is mentioned in the report.
        - If a match is found:
            - Include their name and all keywords from the <FACTS> list related to that person.
            - Ensure that these added keywords are relevant to the context of the report.
        - Do not add any keywords that are not present in the <FACTS> list or the report text.
    5. Deduplicate the extracted keywords.
    6. Check the spelling and correctness of the keywords, ensuring they exactly match the words used in the text or in the <FACTS> list.
    7. Format the result as shown in the <FORMAT> section.
    </STEPS>

    <FORMAT>
    {{
    "keywords": ["keyword1", "keyword2", "keyword3"]
    }}
    </FORMAT>

    <EXAMPLE>
    Input:

    - Facts:
        Anna Nowak: ["oficer bezpieczeństwa", "Kraków", "ewakuacja", "wyciek chemikaliów", "Jan Kowalski", "procedury bezpieczeństwa", "wiedza ekspercka", "reagowanie kryzysowe", "bezpieczeństwo chemiczne"]

    - Report:
        Nazwa raportu: raport-00-Sektor-C4
        W dniu 2022-01-15 w Krakowie odbyła się konferencja na temat bezpieczeństwa chemicznego w fabrykach. W konferencji wzięli udział eksperci z różnych dziedzin, w tym Anna Nowak.
    ---

    Output:
    {{
    "keywords": ["2022-01-15", "Anna Nowak", "bezpieczeństwo chemiczne", "ewakuacja", "fabryki", "Jan Kowalski", "Kraków", "konferencja", "oficer bezpieczeństwa", "procedury bezpieczeństwa", "raport 00", "reagowanie kryzysowe", "Sektor C4", "wiedza ekspercka", "wyciek chemikaliów"]
    }}
    </EXAMPLE>
    """

    system_prompt = system_prompt_template.format(facts=facts)

    message = f"Nazwa raportu: {filename}\n\n{report}"

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        response_format=ReportKeywords
    )

    return completion.choices[0].message.parsed


def json_file_to_pydantic(json_file: Path, model: BaseModel) -> BaseModel:
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    return model.model_validate(json_data)

def pydantic_to_json_file(data: BaseModel, json_file: Path):
    with json_file.open("w", encoding="utf-8") as f:
        f.write(data.model_dump_json())


def extract_facts_from_document(path: Path) -> list[FactKeywords]:
    facts = []

    for fact_file in FACTS_PATH.glob("*.txt"):
        json_file = Path(FACTS_PATH / "json" / fact_file.with_suffix(".json").name)

        if json_file.exists():
            print(f"Fact keywords already extracted. Loading from {json_file.name}")
            # Load data from the JSON file
            data = json_file_to_pydantic(json_file, FactKeywords)
            facts.append(data)
            
        else:
            text = fact_file.read_text(encoding="utf-8")

            print(f"Checking if the document contains any facts: {fact_file.name}")
            # Check if the document contains any facts
            has_facts = contains_facts(text)

            if has_facts:
                print("Document contains facts. Extracting keywords...")

                # Extract the name and keywords for each fact
                data = extract_name_keywords_from_fact(text)
                facts.append(data)

                # Save the extracted name and keywords to a JSON file
                pydantic_to_json_file(data, json_file)

                print(f"Extracted facts keywords from {fact_file.name} and saved to {json_file.name}")

    
    return facts



def facts_to_str(facts: list[FactKeywords]) -> str:
    list_of_facts = []

    for fact in facts:
        fact_str = f"{fact.name}: {', '.join(fact.keywords)}"
        list_of_facts.append(fact_str)

    return "\n".join(list_of_facts)


def extract_keywords_from_reports(facts_str: str) -> dict[str, str]:
    results = {}

    for report_file in REPORTS_PATH.glob("*.txt"):
        report = report_file.read_text(encoding="utf-8")
        report_name = report_file.stem

        # add a folder before the keywords_file
        keywords_file = Path(REPORTS_PATH / "json" / report_file.with_suffix(".json").name)
        print(keywords_file)

        if keywords_file.exists():
            print(f"Keywords already extracted. Loading from {keywords_file.name}")
            # Load data from the JSON file
            keywords = json_file_to_pydantic(keywords_file, ReportKeywords)
            print(f"Keywords loaded from {keywords_file.name}: {keywords.keywords}")
            
        else:
            print(f"Extracting keywords from {report_name}.txt...")

            # Extract keywords from the report
            keywords = extract_keywords_from_report(report_name, report, facts_str)

            print(f"Keywords extracted from {report_name}.txt: {keywords.keywords}")

            # Save the extracted keywords to a JSON file
            pydantic_to_json_file(keywords, keywords_file)

            print(f"Extracted keywords saved to {keywords_file.name}")


        results[report_file.name] = ", ".join(keywords.keywords)    

    return results

if __name__ == "__main__":
    # Extract facts from the documents
    facts = extract_facts_from_document(FACTS_PATH)

    # Convert the facts to a JSON string
    facts_str = facts_to_str(facts)

    # Extract keywords from the reports
    report_keywords = extract_keywords_from_reports(facts_str)

    print(report_keywords)

    # Submit answers to the AIDevs API
    response = aidevs_client.verify_task("dokumenty", report_keywords)
    print(response)
