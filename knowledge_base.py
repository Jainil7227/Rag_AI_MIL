import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import uuid
import sys
sys.path.append('.')

from chunking_utility import TextChunker

class KnowledgeBase:
    """
    Knowledge base using ChromaDB for vector storage and semantic retrieval.
    """
    
    def __init__(self, collection_name: str = "gdg_knowledge"):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            collection_name (str): Name for this knowledge collection
        """
        print(f"Initializing Knowledge Base '{collection_name}'...")
        
        self.client = chromadb.Client()
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "GDG Workshop Knowledge Base"}
        )
        
        self.chunker = TextChunker(chunk_size=500, overlap=50)
        
        print("Knowledge Base ready.")
    
    def add_document(self, text: str, metadata: Dict = None) -> List[str]:
        """
        Add a document to the knowledge base (chunks, embeds, and stores).
        
        Args:
            text (str): The document text
            metadata (dict): Optional metadata
            
        Returns:
            list: IDs of chunks that were added
        """
        if metadata is None:
            metadata = {}
        
        chunks = self.chunker.chunk_text(text, method='sentences')
        
        ids = []
        texts = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            texts.append(chunk['text'])
            
            chunk_metadata = {
                **metadata,
                'chunk_id': chunk['chunk_id'],
                'word_count': chunk['word_count'],
                'method': chunk.get('method', 'unknown')
            }
            metadatas.append(chunk_metadata)
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        return ids
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        Search the knowledge base using semantic similarity.
        
        Args:
            query_text (str): Your search query
            top_k (int): How many results to return
            
        Returns:
            list: Most relevant chunks with metadata
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        formatted_results = []
        
        if results['ids']:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i] if 'distances' in results and results['distances'] else None
                similarity = (1 - distance) if distance is not None else None
                
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': distance,
                    'similarity': similarity
                })
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the knowledge base.
        """
        return {
            'collection_name': self.collection.name,
            'total_chunks': self.collection.count(),
            'embedding_dimension': 384,
            'embedding_model': 'all-MiniLM-L6-v2'
        }
    
    def clear(self):
        """
        Clear all documents from the knowledge base.
        """
        print("Clearing knowledge base...")
        self.client.delete_collection(self.collection.name)
        
        self.collection = self.client.create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function
        )
        print("Knowledge base cleared.")


if __name__ == "__main__":
    print("\n=== KNOWLEDGE BASE DEMO ===")
    
    kb = KnowledgeBase(collection_name="gdg_demo")
    
    gdg_docs = """
    Google Developer Groups (GDG) are community groups for college and university 
    students interested in Google developer technologies. Students from all undergraduate 
    or graduate programs with an interest in growing as a developer are welcome. By 
    joining a GDG, students grow their knowledge in a peer-to-peer learning environment 
    and build solutions for local businesses and their community.
    
    Events and Activities:
    GDG chapters host various events including workshops, hackathons, study jams, and 
    tech talks. These events are designed to help students learn new technologies, 
    network with peers, and gain practical experience. Workshops typically run from 
    9:00 AM to 5:00 PM and cover topics like AI, Cloud Computing, Android Development, 
    and Web Technologies.
    """
    
    kb.add_document(
        gdg_docs,
        metadata={
            'source': 'GDG Guidelines',
            'type': 'official',
            'category': 'documentation'
        }
    )
    
    stats = kb.get_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    
    test_queries = ["How do I join GDG?", "What kind of events does GDG organize?"]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = kb.query(query, top_k=1)
        if results:
            print(f"Result: {results[0]['text'][:100]}...")