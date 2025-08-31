import json
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from config import settings

class KnowledgeGraphManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
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
        
        last_node = query_node
        for i, step in enumerate(plan):
            tool_node = tx.run("MERGE (t:Tool {name: $name}) RETURN t", name=step['name']).single()['t']
            step_node = tx.run("CREATE (s:Step {order: $order}) RETURN s", order=i).single()['s']
            
            tx.run("MATCH (a), (b) WHERE id(a) = $id_a AND id(b) = $id_b CREATE (a)-[:HAS_STEP]->(b)", id_a=last_node.id, id_b=step_node.id)
            tx.run("MATCH (a:Step), (b:Tool) WHERE id(a) = $id_a AND id(b) = $id_b CREATE (a)-[:CALLS]->(b)", id_a=step_node.id, id_b=tool_node.id)
            
            args_str = json.dumps(step.get('parameters', {}))
            tx.run("MATCH (s:Step) WHERE id(s) = $id SET s.arguments = $args", id=step_node.id, args=args_str)
            last_node = step_node

    def find_successful_plan(self, user_query: str):
        embedding = self.encoder.encode(user_query).tolist()
        with self.driver.session() as session:
            result = session.execute_read(self._find_similar_plan, embedding)
        return result

    @staticmethod
    def _find_similar_plan(tx, embedding):
        res = tx.run(
            "CALL db.index.vector.queryNodes('query_embeddings', 1, $embedding) YIELD node, score "
            "WHERE score > 0.85 "
            "MATCH (node)-[:HAS_STEP*]->(step:Step) "
            "MATCH (step)-[:CALLS]->(tool:Tool) "
            "RETURN step.order AS order, tool.name AS tool_name, step.arguments AS args "
            "ORDER BY order",
            embedding=embedding
        )
        plan = []
        for record in res:
            plan.append({
                "name": record["tool_name"],
                "parameters": json.loads(record["args"])
            })
        return plan if plan else None