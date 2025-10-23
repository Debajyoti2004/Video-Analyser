import json
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import config

class KnowledgeGraphManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.ensure_constraints_and_indices()

    def close(self):
        self.driver.close()

    def ensure_constraints_and_indices(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (q:Query) REQUIRE q.text IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tool) REQUIRE t.name IS UNIQUE")
            session.run(
                "CREATE VECTOR INDEX `query_embeddings` IF NOT EXISTS FOR (q:Query) ON (q.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}"
            )
            session.run("CALL db.awaitIndex('query_embeddings')")

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def store_successful_plan(self, user_query: str, plan: list[dict]):
        embedding = self.encoder.encode(user_query).tolist()
        with self.driver.session() as session:
            session.execute_write(self._create_plan_graph, user_query, embedding, plan)

    @staticmethod
    def _create_plan_graph(tx, user_query, embedding, plan):
        query_node = tx.run(
            "MERGE (q:Query {text: $text}) SET q.embedding = $embedding RETURN q",
            text=user_query, embedding=embedding
        ).single()['q']

        last_node_id = query_node.element_id
        for i, step in enumerate(plan):
            tool_node = tx.run("MERGE (t:Tool {name: $name}) RETURN t", name=step['name']).single()['t']
            step_node = tx.run("CREATE (s:Step {order: $order}) RETURN s", order=i).single()['s']

            tx.run("MATCH (a) WHERE elementId(a) = $id_a "
                   "MATCH (b) WHERE elementId(b) = $id_b "
                   "CREATE (a)-[:HAS_STEP]->(b)", id_a=last_node_id, id_b=step_node.element_id)

            tx.run("MATCH (a:Step) WHERE elementId(a) = $id_a "
                   "MATCH (b:Tool) WHERE elementId(b) = $id_b "
                   "CREATE (a)-[:CALLS]->(b)", id_a=step_node.element_id, id_b=tool_node.element_id)

            args_str = json.dumps(step.get('parameters', {}))
            tx.run("MATCH (s:Step) WHERE elementId(s) = $id SET s.arguments = $args", id=step_node.element_id, args=args_str)
            last_node_id = step_node.element_id

    def find_successful_plan(self, user_query: str):
        embedding = self.encoder.encode(user_query).tolist()
        with self.driver.session() as session:
            result = session.execute_read(self._find_similar_plan, embedding)
        return result

    @staticmethod
    def _find_similar_plan(tx, embedding):
        res = tx.run(
            """
            CALL db.index.vector.queryNodes('query_embeddings', 1, $embedding) YIELD node, score
            WHERE score > 0.80
            MATCH (node)-[:HAS_STEP*]->(step:Step)
            MATCH (step)-[:CALLS]->(tool:Tool)
            RETURN DISTINCT step.order AS order, tool.name AS tool_name, step.arguments AS args
            ORDER BY order
            """,
            embedding=embedding
        )
        plan = []
        for record in res:
            plan.append({
                "name": record["tool_name"],
                "parameters": json.loads(record["args"])
            })
        return plan if plan else None
    
if __name__ == '__main__':
    from rich.console import Console
    from rich.panel import Panel
    from rich.pretty import pprint
    from dotenv import load_dotenv

    def test_knowledge_graph():
        load_dotenv()
        console = Console()
        console.print(Panel("[bold yellow]🧠 KnowledgeGraphManager Standalone Test 🧠[/bold yellow]", border_style="yellow"))

        try:
            console.print("\n[bold]Step 1: Initializing KnowledgeGraphManager...[/bold]")
            kg_manager = KnowledgeGraphManager()
            console.print("[green]✅ Initialization complete. Connected to Neo4j and index is online.[/green]")

            console.print("\n[bold]Step 2: Storing a sample successful plan...[/bold]")
            sample_query = "Show me a picture of the car"
            sample_plan = [
                {"name": "get_snapshot_for_analysis", "parameters": {"video_path": "test_video.mp4", "timestamp": 12.5}},
                {"name": "create_annotated_visual_evidence", "parameters": {"image_path": "snapshots/snapshot_at_12.50s.jpg", "annotations": [{"box": [100, 100, 200, 200], "label": "car"}]}}
            ]
            kg_manager.store_successful_plan(sample_query, sample_plan)
            console.print(f"[green]✅ Stored plan for query:[/green] '{sample_query}'")
            pprint(sample_plan)

            console.print("\n[bold]Step 3: Retrieving plan for a semantically similar query...[/bold]")
            similar_query = "Create a visual of the automobile"
            console.print(f"Searching for plan similar to: '{similar_query}'")
            retrieved_plan = kg_manager.find_successful_plan(similar_query)
            
            if retrieved_plan:
                console.print("[green]✅ Successfully retrieved a similar plan:[/green]")
                pprint(retrieved_plan)
            else:
                console.print("[bold red]❌ Failed to retrieve a similar plan. This might happen on the very first run. Try running the test again.[/bold red]")

            kg_manager.close()

        except Exception as e:
            console.print(Panel(f"[bold red]An error occurred during the test:[/bold red]\n{e}", border_style="red"))
            console.print("[yellow]NOTE: Ensure your Neo4j Docker container is running and the .env file is configured correctly.[/yellow]")

        console.print(Panel("[bold green]🏁 Test complete 🏁[/bold green]", border_style="green"))

    test_knowledge_graph()