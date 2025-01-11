import os
from dotenv import load_dotenv
from icecream import ic
from openai import OpenAI
from pydantic import BaseModel
from utils.client import AIDevsClient


class SQLAgentResponse(BaseModel):
    thoughts: list[str]
    sql_query: str
    answer: str


# Load environment variables from .env file
load_dotenv()


# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
aidevs_client = AIDevsClient(
    api_key=os.getenv("AIDEVS_API_KEY"),
    base_url="https://centrala.ag3nts.org/",
    verify_dir="report",
)











TASK_NAME = "database"

response = aidevs_client.query_db(
    query="SELECT * FROM weapons WHERE weapon_type='rifle';",
    task=TASK_NAME,
)

ic(response)



def run_sql_agent(user_query: str, task_name: str) -> str:
    system_prompt = """
    Your task is to generate a SQL query based on the user's input. The user input is a question or task that should be answered by the database.
    Since you do not know the underlying database schema, as a first step, you should figure out what information you need.
    For now, you have access to the following commands:
    1. "SHOW TABLES" - (non-standard SQL command) to list available tables in the database.
    2. "SHOW CREATE TABLE <table_name>" - (non-standard SQL command) to show the schema of a specific table.
    3. Standard SQL "SELECT" queries (including ordering, filtering, and joining tables).

    Before generating the SQL query, you should analyze the user input to understand the information needed from the database.
    Write down your current thoughts/action plan (step by step) to solve the task as well as the SQL query you would like to execute.

    <RULES>
    1. If you do not have enough information to generate the SQL query, use the "SHOW TABLES" or "SHOW CREATE TABLE <table_name>" commands to gather more information.
    2. For the final answer, return only the requested information e.g. a list of IDs, names, or values - DO NOT return any comments or additional information.
    </RULES>

    Return your thoughts and the SQL query in the following <FORMAT>:
    <FORMAT>
    {
        "thoughts": [A list of your thoughts and actions],
        "sql_query": "Your SQL query here",
        "answer": Empty string or the answer to the user query,
    }
    </FORMAT>
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    for i in range(20):
        ic(messages)
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=SQLAgentResponse,
        )

        sql_response = completion.choices[0].message.parsed


        if sql_response.answer:
            return sql_response.answer


        assistant_msg = f"""
        Thoughts:
        {"\n".join(sql_response.thoughts)}

        SQL Query:
        {sql_response.sql_query}
        """

        ic(assistant_msg)

        db_response = aidevs_client.query_db(
            query=sql_response.sql_query,
            task=task_name,
        )

        user_msg = f"""
        Database Response:
        {db_response.reply}

        Error:
        {db_response.error}
        """

        ic(user_msg)

        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": user_msg})


        input("Press Enter to continue...")


answer = run_sql_agent("Which active datacenters (DC_ID) are managed by users that are on vacation (is_active=0). Return them as a list of DC_ID.", TASK_NAME)
ic(answer)

ids = answer.split(", ")
ic(ids)

# Submit answer to the AIDevs API
response = aidevs_client.verify_task("database", ids)
ic(response)

