
import os

from dotenv import load_dotenv
from icecream import ic
from neo4j import GraphDatabase
from utils.client import AIDevsClient

# Initialize environment and clients
load_dotenv()

aidevs_client = AIDevsClient(
    api_key=os.getenv("AIDEVS_API_KEY"),
    base_url="https://centrala.ag3nts.org/",
    verify_dir="report",
)
neo4j_driver = GraphDatabase.driver(
    uri=os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

TASK_NAME = "connections"


def create_graph(tx, connections):
    """Create user nodes and relationships in Neo4j."""
    query = """
    UNWIND $connections as conn
    MERGE (u1:User {username: conn.user1})
    MERGE (u2:User {username: conn.user2})
    MERGE (u1)-[r:CONNECTED_TO]->(u2)
    """
    tx.run(query, connections=connections)


def run_shortest_path(tx, user1, user2):
    """Find shortest path between two users."""
    query = """
    MATCH (user1:User {username: $user1})
    MATCH (user2:User {username: $user2})
    MATCH path = SHORTESTPATH((user1)-[*]-(user2))
    RETURN [n IN nodes(path) | n.username] as path_names, 
           length(path) as path_length
    """
    return tx.run(query, user1=user1, user2=user2).data()


def find_shortest_path(user1, user2):
    """Wrapper to execute shortest path query."""
    with neo4j_driver.session() as session:
        return session.execute_read(run_shortest_path, user1, user2)


def init_graph(connections):
    """Initialize graph with connection data."""
    with neo4j_driver.session() as session:
        session.execute_write(create_graph, connections)


if __name__ == "__main__":
    
    # SQL query to get user connections
    db_query = """
    SELECT 
        u1.username as user1,
        u2.username as user2
    FROM connections c
    JOIN users u1 ON c.user1_id = u1.id
    JOIN users u2 ON c.user2_id = u2.id
    """

    # Main execution
    response = aidevs_client.query_db(query=db_query, task="database")
    ic(response.reply)

    init_graph(response.reply)

    result = find_shortest_path("Rafał", "Barbara")
    ic(result)

    # Format and verify result
    name_str = ", ".join(result[0]["path_names"])
    response = aidevs_client.verify_task(TASK_NAME, name_str)
    ic(response)
