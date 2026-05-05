"""
GraphRAG vs Flat RAG - Tech Company Corpus
Lab Day 19: Xay dung he thong GraphRAG

Cac buoc chay:
1. pip install -r requirements.txt
2. Copy .env.example thanh .env va dien API key
3. python main.py
"""

import json
import os
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Any

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
# PHAN 1: DATA LOADING
# ============================================================================

def load_corpus(path: str = "corpus.json") -> List[Dict]:
    """Load corpus text tu file JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_triples(path: str = "triples.json") -> List[Dict]:
    """Load triples (head, relation, tail) tu file JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# PHAN 2: FLAT RAG - Vector Search
# ============================================================================

class FlatRAG:
    """Flat RAG: Chia text thanh chunks -> Embedding -> Vector search."""

    def __init__(self, corpus: List[Dict], chunk_size: int = 500, chunk_overlap: int = 100):
        self.corpus = corpus
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        self.embeddings: np.ndarray = None
        self._build_chunks()

    def _build_chunks(self):
        """Chia van ban thanh cac chunks."""
        for doc in self.corpus:
            text = doc["text"]
            company = doc["company"]
            # Simple chunking by words
            words = text.split()
            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                self.chunks.append(chunk_text)
                self.chunk_metadata.append({"company": company, "start": start, "end": end})
                start += self.chunk_size - self.chunk_overlap

    def build_embeddings(self):
        """Tao embedding cho tat ca chunks bang OpenAI API."""
        print(f"[FlatRAG] Dang tao embeddings cho {len(self.chunks)} chunks...")
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
            print(f"[FlatRAG] Da xu ly {min(i + batch_size, len(self.chunks))}/{len(self.chunks)} chunks")
        self.embeddings = np.array(all_embeddings)
        print("[FlatRAG] Hoan thanh tao embeddings!")

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Tim kiem cac chunks lien quan nhat den query."""
        # Tao embedding cho query
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        query_embedding = np.array([response.data[0].embedding])

        # Tinh cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Lay top-k chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_chunks = [self.chunks[i] for i in top_indices]
        retrieved_scores = [float(similarities[i]) for i in top_indices]

        return retrieved_chunks, retrieved_scores

    def answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Tra loi cau hoi bang Flat RAG."""
        start_time = time.time()

        # Retrieve
        chunks, scores = self.retrieve(query, top_k=top_k)
        context = "\n\n---\n\n".join(chunks)

        # Generate answer
        prompt = f"""Tra loi cau hoi dua tren thong tin duoi day. Neu khong co du thong tin, hay noi ro.

THONG TIN:
{context}

CAU HOI: {query}

TRA LOI:"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Ban la tro ly AI chuyen tra loi cau hoi dua tren tai lieu."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        elapsed = time.time() - start_time

        # Tinh token
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
# PHAN 3: GRAPH RAG - Knowledge Graph + Multi-hop Query
# ============================================================================

class GraphRAG:
    """Graph RAG: Xay dung Knowledge Graph -> Entity extraction -> Graph traversal -> LLM."""

    def __init__(self, triples: List[Dict]):
        self.triples = triples
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """Xay dung do thi tu cac triples."""
        print("[GraphRAG] Dang xay dung Knowledge Graph...")
        for triple in self.triples:
            head = triple["head"]
            relation = triple["relation"]
            tail = triple["tail"]
            source = triple.get("source", "")

            # Add nodes voi thuoc tinh
            if head not in self.graph:
                self.graph.add_node(head, type="entity", sources=[])
            if tail not in self.graph:
                self.graph.add_node(tail, type="entity", sources=[])

            # Add source vao nodes
            if source and source not in self.graph.nodes[head].get("sources", []):
                self.graph.nodes[head].setdefault("sources", []).append(source)
            if source and source not in self.graph.nodes[tail].get("sources", []):
                self.graph.nodes[tail].setdefault("sources", []).append(source)

            # Add edge
            self.graph.add_edge(head, tail, relation=relation, source=source)

        print(f"[GraphRAG] Graph hoan thanh: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def extract_entities_from_query(self, query: str) -> List[str]:
        """Trich xuat entity tu query bang LLM."""
        prompt = f"""Trich xuat cac ten thuc the (entity) lien quan den cong ty, nguoi, san pham, hoac dia diem tu cau hoi sau.
Chi tra ve danh sach cac entity, moi entity tren mot dong.

CAU HOI: {query}

CAC ENTITY:"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Ban la he thong trich xuat thuc the. Chi tra ve danh sach entity."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        entities_text = response.choices[0].message.content.strip()
        entities = [e.strip() for e in entities_text.split("\n") if e.strip()]
        return entities

    def fuzzy_match_entity(self, entity: str) -> str:
        """Tim node trong graph khop gan dung voi entity."""
        # Exact match
        if entity in self.graph:
            return entity

        # Case-insensitive match
        for node in self.graph.nodes():
            if node.lower() == entity.lower():
                return node

        # Partial match
        for node in self.graph.nodes():
            if entity.lower() in node.lower() or node.lower() in entity.lower():
                return node

        return None

    def traverse(self, start_entity: str, max_hops: int = 2) -> List[Dict]:
        """Duyet do thi tu entity bat dau, tra ve cac path trong max_hops buoc."""
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

            # Add current path info
            if hops > 0:
                paths.append({
                    "path": path,
                    "entity": current,
                    "hops": hops
                })

            # Explore neighbors
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
        """Lay context tu subgraph xung quanh cac entity."""
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

        # Also get direct neighbors
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
        """Tra loi cau hoi bang Graph RAG."""
        start_time = time.time()

        # Step 1: Extract entities from query
        entities = self.extract_entities_from_query(query)
        print(f"[GraphRAG] Entities trich xuat: {entities}")

        # Step 2: Traverse graph
        graph_context = self.get_subgraph_context(entities, max_hops=max_hops)
        print(f"[GraphRAG] So triples tim duoc: {len(graph_context.split(chr(10))) if graph_context else 0}")

        # Step 3: Generate answer
        prompt = f"""Tra loi cau hoi dua tren cac quan he (triples) tu Knowledge Graph duoi day.
Moi quan he co dang: (Entity1, Relation, Entity2)
Neu khong co du thong tin, hay noi ro.

CAC QUAN HE TRONG KNOWLEDGE GRAPH:
{graph_context if graph_context else "Khong tim thay quan he lien quan."}

CAU HOI: {query}

TRA LOI:"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Ban la tro ly AI chuyen tra loi cau hoi dua tren Knowledge Graph."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        elapsed = time.time() - start_time

        # Tinh token
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
# PHAN 4: VISUALIZATION
# ============================================================================

def visualize_graph(graph: nx.DiGraph, output_path: str = "graph_visualization.png", max_nodes: int = 50):
    """Ve do thi va luu thanh file anh."""
    import matplotlib.pyplot as plt

    # Lay subgraph voi so node gioi han
    if len(graph.nodes()) > max_nodes:
        nodes = list(graph.nodes())[:max_nodes]
        subgraph = graph.subgraph(nodes).copy()
    else:
        subgraph = graph

    plt.figure(figsize=(20, 16))
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

    # Ve nodes
    node_colors = []
    for node in subgraph.nodes():
        sources = subgraph.nodes[node].get("sources", [])
        if sources:
            node_colors.append("skyblue")
        else:
            node_colors.append("lightgreen")

    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight="bold")

    # Ve edges
    nx.draw_networkx_edges(subgraph, pos, edge_color="gray", arrows=True,
                           arrowsize=15, arrowstyle="->", width=1.5,
                           connectionstyle="arc3,rad=0.1")

    # Ve edge labels
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
    print(f"[Visualization] Da luu do thi tai: {output_path}")


# ============================================================================
# PHAN 5: BENCHMARK - 20 CAU HOI
# ============================================================================

BENCHMARK_QUESTIONS = [
    # Single-hop (don gian)
    "OpenAI duoc thanh lap nam nao?",
    "Ai la nguoi sang lap Google?",
    "Microsoft duoc thanh lap boi ai?",
    "Meta Platforms so huu nhung nen tang nao?",
    "Apple Inc. duoc thanh lap nam nao?",
    "Amazon duoc thanh lap boi ai?",
    "Tesla duoc thanh lap nam nao?",
    "Nvidia duoc thanh lap boi ai?",
    "Samsung Electronics duoc thanh lap nam nao?",
    "Intel duoc thanh lap boi ai?",

    # Multi-hop (phuc tap - can nhieu buoc suy luan)
    "CEO cua cong ty duoc Elon Musk dong sang lap la ai?",
    "Sam Altman co vai tro gi tai OpenAI?",
    "Sundar Pichai la CEO cua cong ty nao?",
    "Microsoft da mua lai cong ty nao trong linh vuc mang xa hoi?",
    "Apple da mua lai cong ty nao de dua Steve Jobs tro lai?",
    "Amazon da mua lai chuoi sieu thi nao voi gia bao nhieu?",
    "Elon Musk la CEO cua cong ty nao ngoai Tesla?",
    "Nvidia da dat gia tri von hoa bao nhieu vao nam 2025?",
    "Google co nhung san pham nao?",
    "Tim Cook la CEO cua cong ty nao va nguoi ke nhiem ong ay la ai?",
]


def run_benchmark(flat_rag: FlatRAG, graph_rag: GraphRAG, questions: List[str] = None):
    """Chay benchmark so sanh Flat RAG va Graph RAG."""
    if questions is None:
        questions = BENCHMARK_QUESTIONS

    results = []

    print("\n" + "=" * 80)
    print("BENCHMARK: FLAT RAG vs GRAPH RAG")
    print("=" * 80)

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"CAU HOI {i}/20: {question}")
        print("=" * 60)

        # Flat RAG
        print("\n--- Flat RAG ---")
        try:
            flat_result = flat_rag.answer(question)
            print(f"Tra loi: {flat_result['answer'][:200]}...")
            print(f"Tokens: {flat_result['total_tokens']} | Thoi gian: {flat_result['elapsed_time']:.2f}s")
        except Exception as e:
            print(f"Loi Flat RAG: {e}")
            flat_result = {"method": "FlatRAG", "query": question, "answer": f"ERROR: {e}",
                           "total_tokens": 0, "elapsed_time": 0}

        # Graph RAG
        print("\n--- Graph RAG ---")
        try:
            graph_result = graph_rag.answer(question)
            print(f"Tra loi: {graph_result['answer'][:200]}...")
            print(f"Tokens: {graph_result['total_tokens']} | Thoi gian: {graph_result['elapsed_time']:.2f}s")
        except Exception as e:
            print(f"Loi Graph RAG: {e}")
            graph_result = {"method": "GraphRAG", "query": question, "answer": f"ERROR: {e}",
                            "total_tokens": 0, "elapsed_time": 0}

        results.append({
            "question": question,
            "flat_rag": flat_result,
            "graph_rag": graph_result
        })

    return results


def print_benchmark_summary(results: List[Dict]):
    """In tom tat ket qua benchmark."""
    print("\n" + "=" * 80)
    print("TOM TAT BENCHMARK")
    print("=" * 80)

    flat_total_tokens = sum(r["flat_rag"]["total_tokens"] for r in results)
    graph_total_tokens = sum(r["graph_rag"]["total_tokens"] for r in results)
    flat_total_time = sum(r["flat_rag"]["elapsed_time"] for r in results)
    graph_total_time = sum(r["graph_rag"]["elapsed_time"] for r in results)

    print(f"\nTong so cau hoi: {len(results)}")
    print(f"\n--- Flat RAG ---")
    print(f"  Tong tokens: {flat_total_tokens}")
    print(f"  Tong thoi gian: {flat_total_time:.2f}s")
    print(f"  Trung binh tokens/cau: {flat_total_tokens / len(results):.0f}")
    print(f"  Trung binh thoi gian/cau: {flat_total_time / len(results):.2f}s")

    print(f"\n--- Graph RAG ---")
    print(f"  Tong tokens: {graph_total_tokens}")
    print(f"  Tong thoi gian: {graph_total_time:.2f}s")
    print(f"  Trung binh tokens/cau: {graph_total_tokens / len(results):.0f}")
    print(f"  Trung binh thoi gian/cau: {graph_total_time / len(results):.2f}s")

    print(f"\n--- So sanh ---")
    print(f"  Chenh lech tokens: {graph_total_tokens - flat_total_tokens:+,}")
    print(f"  Chenh lech thoi gian: {graph_total_time - flat_total_time:+.2f}s")

    # Luu ket qua
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n[Benchmark] Da luu ket qua chi tiet tai: benchmark_results.json")


# ============================================================================
# PHAN 6: MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("GRAPH RAG vs FLAT RAG - TECH COMPANY CORPUS")
    print("=" * 80)

    # 1. Load data
    print("\n[1] Dang tai du lieu...")
    corpus = load_corpus("corpus.json")
    triples = load_triples("triples.json")
    print(f"  Corpus: {len(corpus)} cong ty")
    print(f"  Triples: {len(triples)} quan he")

    # 2. Initialize Flat RAG
    print("\n[2] Khoi tao Flat RAG...")
    flat_rag = FlatRAG(corpus, chunk_size=400, chunk_overlap=50)
    flat_rag.build_embeddings()

    # 3. Initialize Graph RAG
    print("\n[3] Khoi tao Graph RAG...")
    graph_rag = GraphRAG(triples)

    # 4. Visualize graph
    print("\n[4] Dang ve do thi...")
    visualize_graph(graph_rag.graph, output_path="graph_visualization.png", max_nodes=60)

    # 5. Demo single question
    print("\n[5] Demo voi mot cau hoi mau...")
    demo_question = "CEO cua cong ty duoc Elon Musk dong sang lap la ai?"
    print(f"\nCau hoi: {demo_question}")

    print("\n--- Flat RAG ---")
    flat_result = flat_rag.answer(demo_question)
    print(f"Tra loi: {flat_result['answer']}")

    print("\n--- Graph RAG ---")
    graph_result = graph_rag.answer(demo_question)
    print(f"Tra loi: {graph_result['answer']}")

    # 6. Run benchmark
    print("\n[6] Chay benchmark 20 cau hoi...")
    benchmark_results = run_benchmark(flat_rag, graph_rag)
    print_benchmark_summary(benchmark_results)

    print("\n" + "=" * 80)
    print("HOAN THANH!")
    print("=" * 80)
    print("\nCac file da tao:")
    print("  - graph_visualization.png: Hinh anh do thi")
    print("  - benchmark_results.json: Ket qua benchmark chi tiet")


if __name__ == "__main__":
    main()
