"""
GraphRAG vs Flat RAG - Tech Company Corpus
Lab Day 19: Building GraphRAG System

Steps to run:
1. pip install -r requirements.txt
2. Copy .env.example to .env and fill in API key
3. python main.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# Fix Unicode output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

def load_corpus(path: str = "corpus.json") -> List[Dict]:
    """Load corpus text from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_triples(path: str = "triples.json") -> List[Dict]:
    """Load triples (head, relation, tail) from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# PART 2: FLAT RAG - Vector Search
# ============================================================================

class FlatRAG:
    """Flat RAG: Split text into chunks -> Embedding -> Vector search."""

    def __init__(self, corpus: List[Dict], chunk_size: int = 500, chunk_overlap: int = 100):
        self.corpus = corpus
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        self.embeddings: np.ndarray = None
        self._build_chunks()

    def _build_chunks(self):
        """Split text into chunks."""
        for doc in self.corpus:
            text = doc["text"]
            company = doc["company"]
            words = text.split()
            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                self.chunks.append(chunk_text)
                self.chunk_metadata.append({"company": company, "start": start, "end": end})
                start += self.chunk_size - self.chunk_overlap

    def build_embeddings(self):
        """Create embeddings for all chunks using OpenAI API."""
        print(f"[FlatRAG] Creating embeddings for {len(self.chunks)} chunks...")
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            print(f"[FlatRAG] Processed {min(i + batch_size, len(self.chunks))}/{len(self.chunks)} chunks")
        self.embeddings = np.array(all_embeddings)
        print("[FlatRAG] Embeddings created successfully!")

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Search for the most relevant chunks for the query."""
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        query_embedding = np.array([response.data[0].embedding])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_chunks = [self.chunks[i] for i in top_indices]
        retrieved_scores = [float(similarities[i]) for i in top_indices]
        return retrieved_chunks, retrieved_scores

    def answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer question using Flat RAG."""
        start_time = time.time()
        chunks, scores = self.retrieve(query, top_k=top_k)
        context = "\n\n---\n\n".join(chunks)

        prompt = f"""Answer the question based on the information below. If there is not enough information, say so clearly.

INFORMATION:
{context}

QUESTION: {query}

ANSWER:"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in answering questions based on documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        elapsed = time.time() - start_time
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        return {
            "method": "FlatRAG",
            "query": query,
            "answer": answer,
            "retrieved_chunks": chunks,
            "retrieved_scores": scores,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "elapsed_time": elapsed
        }


# ============================================================================
# PART 3: GRAPH RAG - Knowledge Graph + Multi-hop Query
# ============================================================================

class GraphRAG:
    """Graph RAG: Build Knowledge Graph -> Entity extraction -> Graph traversal -> LLM."""

    def __init__(self, triples: List[Dict]):
        self.triples = triples
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """Build graph from triples."""
        print("[GraphRAG] Building Knowledge Graph...")
        for triple in self.triples:
            head = triple["head"]
            relation = triple["relation"]
            tail = triple["tail"]
            source = triple.get("source", "")

            if head not in self.graph:
                self.graph.add_node(head, type="entity", sources=[])
            if tail not in self.graph:
                self.graph.add_node(tail, type="entity", sources=[])

            if source and source not in self.graph.nodes[head].get("sources", []):
                self.graph.nodes[head].setdefault("sources", []).append(source)
            if source and source not in self.graph.nodes[tail].get("sources", []):
                self.graph.nodes[tail].setdefault("sources", []).append(source)

            self.graph.add_edge(head, tail, relation=relation, source=source)

        print(f"[GraphRAG] Graph completed: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from query using LLM."""
        all_nodes = list(self.graph.nodes())
        important_nodes = [n for n in all_nodes if len(n) > 2 and not n.isdigit()]
        node_sample = important_nodes[:100]
        node_list_str = ", ".join(node_sample[:50])

        prompt = f"""Extract PROPER NOUNS (names of companies, people, products, places) from the question below.
Return ONLY proper nouns, one per line.
DO NOT return common words like "company", "CEO", "person", "year", "price".

SOME NAMES IN THE SYSTEM: {node_list_str}

QUESTION: {query}

PROPER NOUNS (only proper nouns, one per line):"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an entity extraction system. Return only proper nouns, one per line. Do not return common words."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        entities_text = response.choices[0].message.content.strip()
        entities = []
        for e in entities_text.split("\n"):
            e = e.strip().strip("-•*1234567890. ")
            common_words = {"company", "ceo", "person", "people", "year", "price", "information",
                            "stock", "market cap", "product", "platform", "chain", "store",
                            "social media", "field", "role", "successor", "none", "who", "what",
                            "when", "where", "how", "which"}
            if e and len(e) > 2 and e.lower() not in common_words:
                entities.append(e)
        return entities

    def fuzzy_match_entity(self, entity: str) -> str:
        """Find node in graph that approximately matches the entity."""
        if entity in self.graph:
            return entity
        for node in self.graph.nodes():
            if node.lower() == entity.lower():
                return node
        for node in self.graph.nodes():
            if entity.lower() in node.lower() or node.lower() in entity.lower():
                return node
        return None

    def traverse(self, start_entity: str, max_hops: int = 2) -> List[Dict]:
        """Traverse graph from starting entity, return paths within max_hops steps."""
        matched = self.fuzzy_match_entity(start_entity)
        if not matched:
            return []

        paths = []
        visited = set()
        queue = [(matched, 0, [])]

        while queue:
            current, hops, path = queue.pop(0)
            if hops > max_hops:
                continue
            if current in visited and hops > 0:
                continue
            visited.add(current)

            if hops > 0:
                paths.append({
                    "path": path,
                    "entity": current,
                    "hops": hops
                })

            for neighbor in self.graph.successors(current):
                edge_data = self.graph.get_edge_data(current, neighbor)
                relation = edge_data.get("relation", "")
                new_path = path + [(current, relation, neighbor)]
                queue.append((neighbor, hops + 1, new_path))

            for neighbor in self.graph.predecessors(current):
                edge_data = self.graph.get_edge_data(neighbor, current)
                relation = edge_data.get("relation", "")
                new_path = path + [(neighbor, relation, current)]
                queue.append((neighbor, hops + 1, new_path))

        return paths

    def get_subgraph_context(self, entities: List[str], max_hops: int = 2) -> str:
        """Get context from subgraph around the entities."""
        all_triples = []
        seen = set()

        for entity in entities:
            paths = self.traverse(entity, max_hops=max_hops)
            for path_info in paths:
                for head, relation, tail in path_info["path"]:
                    triple_key = (head, relation, tail)
                    if triple_key not in seen:
                        seen.add(triple_key)
                        all_triples.append(f"({head}, {relation}, {tail})")

        for entity in entities:
            matched = self.fuzzy_match_entity(entity)
            if matched and matched in self.graph:
                for neighbor in self.graph.successors(matched):
                    edge_data = self.graph.get_edge_data(matched, neighbor)
                    relation = edge_data.get("relation", "")
                    triple_key = (matched, relation, neighbor)
                    if triple_key not in seen:
                        seen.add(triple_key)
                        all_triples.append(f"({matched}, {relation}, {neighbor})")

                for neighbor in self.graph.predecessors(matched):
                    edge_data = self.graph.get_edge_data(neighbor, matched)
                    relation = edge_data.get("relation", "")
                    triple_key = (neighbor, relation, matched)
                    if triple_key not in seen:
                        seen.add(triple_key)
                        all_triples.append(f"({neighbor}, {relation}, {matched})")

        return "\n".join(all_triples)

    def answer(self, query: str, max_hops: int = 2) -> Dict[str, Any]:
        """Answer question using Graph RAG."""
        start_time = time.time()

        entities = self.extract_entities_from_query(query)
        print(f"[GraphRAG] Extracted entities: {entities}")

        graph_context = self.get_subgraph_context(entities, max_hops=max_hops)
        print(f"[GraphRAG] Triples found: {len(graph_context.split(chr(10))) if graph_context else 0}")

        prompt = f"""Answer the question based on the relationships (triples) from the Knowledge Graph below.
Each relationship has the form: (Entity1, Relation, Entity2)
If there is not enough information, say so clearly.

RELATIONSHIPS IN KNOWLEDGE GRAPH:
{graph_context if graph_context else "No relevant relationships found."}

QUESTION: {query}

ANSWER:"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in answering questions based on Knowledge Graphs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        elapsed = time.time() - start_time
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        return {
            "method": "GraphRAG",
            "query": query,
            "answer": answer,
            "extracted_entities": entities,
            "graph_context": graph_context,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "elapsed_time": elapsed
        }


# ============================================================================
# PART 4: VISUALIZATION
# ============================================================================

def visualize_graph(graph: nx.DiGraph, output_path: str = "graph_visualization.png", max_nodes: int = 50):
    """Draw graph and save as image file."""
    import matplotlib.pyplot as plt

    if len(graph.nodes()) > max_nodes:
        nodes = list(graph.nodes())[:max_nodes]
        subgraph = graph.subgraph(nodes).copy()
    else:
        subgraph = graph

    plt.figure(figsize=(20, 16))
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

    node_colors = []
    for node in subgraph.nodes():
        sources = subgraph.nodes[node].get("sources", [])
        if sources:
            node_colors.append("skyblue")
        else:
            node_colors.append("lightgreen")

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(subgraph, pos, edge_color="gray", arrows=True,
                           arrowsize=15, arrowstyle="->", width=1.5,
                           connectionstyle="arc3,rad=0.1")

    edge_labels = {}
    for u, v, data in subgraph.edges(data=True):
        relation = data.get("relation", "")
        edge_labels[(u, v)] = relation

    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=6)

    plt.title("Knowledge Graph - Tech Companies", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Graph saved to: {output_path}")


# ============================================================================
# PART 5: BENCHMARK - 20 QUESTIONS
# ============================================================================

BENCHMARK_QUESTIONS = [
    # Single-hop (simple)
    "When was OpenAI founded?",
    "Who founded Google?",
    "Who founded Microsoft?",
    "What platforms does Meta Platforms own?",
    "When was Apple Inc. founded?",
    "Who founded Amazon?",
    "When was Tesla founded?",
    "Who founded Nvidia?",
    "When was Samsung Electronics founded?",
    "Who founded Intel?",

    # Multi-hop (complex - requires multi-step reasoning)
    "Who is the CEO of the company co-founded by Elon Musk?",
    "What role does Sam Altman have at OpenAI?",
    "Which company is Sundar Pichai the CEO of?",
    "Which social media company did Microsoft acquire?",
    "Which company did Apple acquire to bring Steve Jobs back?",
    "Which supermarket chain did Amazon acquire and for how much?",
    "Besides Tesla, which other company is Elon Musk the CEO of?",
    "What market capitalization did Nvidia reach in 2025?",
    "What products does Google have?",
    "Which company is Tim Cook the CEO of and who is his successor?",
]


def run_benchmark(flat_rag: FlatRAG, graph_rag: GraphRAG, questions: List[str] = None):
    """Run benchmark comparing Flat RAG and Graph RAG."""
    if questions is None:
        questions = BENCHMARK_QUESTIONS

    results = []

    print("\n" + "=" * 80)
    print("BENCHMARK: FLAT RAG vs GRAPH RAG")
    print("=" * 80)

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"QUESTION {i}/20: {question}")
        print("=" * 60)

        print("\n--- Flat RAG ---")
        try:
            flat_result = flat_rag.answer(question)
            print(f"Answer: {flat_result['answer'][:200]}...")
            print(f"Tokens: {flat_result['total_tokens']} | Time: {flat_result['elapsed_time']:.2f}s")
        except Exception as e:
            print(f"Flat RAG Error: {e}")
            flat_result = {"method": "FlatRAG", "query": question, "answer": f"ERROR: {e}",
                           "total_tokens": 0, "elapsed_time": 0}

        print("\n--- Graph RAG ---")
        try:
            graph_result = graph_rag.answer(question)
            print(f"Answer: {graph_result['answer'][:200]}...")
            print(f"Tokens: {graph_result['total_tokens']} | Time: {graph_result['elapsed_time']:.2f}s")
        except Exception as e:
            print(f"Graph RAG Error: {e}")
            graph_result = {"method": "GraphRAG", "query": question, "answer": f"ERROR: {e}",
                            "total_tokens": 0, "elapsed_time": 0}

        results.append({
            "question": question,
            "flat_rag": flat_result,
            "graph_rag": graph_result
        })

    return results


def print_benchmark_summary(results: List[Dict]):
    """Print benchmark summary."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    flat_total_tokens = sum(r["flat_rag"]["total_tokens"] for r in results)
    graph_total_tokens = sum(r["graph_rag"]["total_tokens"] for r in results)
    flat_total_time = sum(r["flat_rag"]["elapsed_time"] for r in results)
    graph_total_time = sum(r["graph_rag"]["elapsed_time"] for r in results)

    print(f"\nTotal questions: {len(results)}")
    print(f"\n--- Flat RAG ---")
    print(f"  Total tokens: {flat_total_tokens}")
    print(f"  Total time: {flat_total_time:.2f}s")
    print(f"  Avg tokens/question: {flat_total_tokens / len(results):.0f}")
    print(f"  Avg time/question: {flat_total_time / len(results):.2f}s")

    print(f"\n--- Graph RAG ---")
    print(f"  Total tokens: {graph_total_tokens}")
    print(f"  Total time: {graph_total_time:.2f}s")
    print(f"  Avg tokens/question: {graph_total_tokens / len(results):.0f}")
    print(f"  Avg time/question: {graph_total_time / len(results):.2f}s")

    print(f"\n--- Comparison ---")
    print(f"  Token difference: {graph_total_tokens - flat_total_tokens:+,}")
    print(f"  Time difference: {graph_total_time - flat_total_time:+.2f}s")

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n[Benchmark] Detailed results saved to: benchmark_results.json")


# ============================================================================
# PART 6: MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("GRAPH RAG vs FLAT RAG - TECH COMPANY CORPUS")
    print("=" * 80)

    print("\n[1] Loading data...")
    corpus = load_corpus("corpus.json")
    triples = load_triples("triples.json")
    print(f"  Corpus: {len(corpus)} companies")
    print(f"  Triples: {len(triples)} relations")

    print("\n[2] Initializing Flat RAG...")
    flat_rag = FlatRAG(corpus, chunk_size=400, chunk_overlap=50)
    flat_rag.build_embeddings()

    print("\n[3] Initializing Graph RAG...")
    graph_rag = GraphRAG(triples)

    print("\n[4] Visualizing graph...")
    visualize_graph(graph_rag.graph, output_path="graph_visualization.png", max_nodes=60)

    print("\n[5] Demo with a sample question...")
    demo_question = "Who is the CEO of the company co-founded by Elon Musk?"
    print(f"\nQuestion: {demo_question}")

    print("\n--- Flat RAG ---")
    flat_result = flat_rag.answer(demo_question)
    print(f"Answer: {flat_result['answer']}")

    print("\n--- Graph RAG ---")
    graph_result = graph_rag.answer(demo_question)
    print(f"Answer: {graph_result['answer']}")

    print("\n[6] Running benchmark with 20 questions...")
    benchmark_results = run_benchmark(flat_rag, graph_rag)
    print_benchmark_summary(benchmark_results)

    print("\n" + "=" * 80)
    print("COMPLETED!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - graph_visualization.png: Graph visualization image")
    print("  - benchmark_results.json: Detailed benchmark results")


if __name__ == "__main__":
    main()
