import sys
import os
from typing import List, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'DAY_2'))

from gemini_wrapper import GeminiWrapper
from knowledge_base import KnowledgeBase

class RAGAgent:
    """
    Intelligent RAG-powered assistant that combines a vector database (Knowledge Base)
    with a Large Language Model (Gemini) to provide accurate, sourced answers.
    """
    
    def __init__(
        self,
        gemini_api_key: str,
        knowledge_base: KnowledgeBase = None,
        temperature: float = 0.3
    ):
        """
        Initialize the RAG Agent.
        
        Args:
            gemini_api_key (str): Your Gemini API key
            knowledge_base (KnowledgeBase): Pre-built knowledge base
            temperature (float): Lower = more factual (0.0-1.0)
        """
        print("Initializing RAG Agent...")
        
        self.llm = GeminiWrapper(
            api_key=gemini_api_key,
            model_name="gemini-2.5-flash",
            temperature=temperature
        )
        
        self.llm.set_persona(
            "You are a helpful AI assistant with access to a knowledge base. "
            "When answering questions, you ALWAYS cite the source documents you used. "
            "If you don't find relevant information in the knowledge base, you say so honestly. "
            "You are accurate, helpful, and always provide context from the documents. "
            "You never make up information - you only use what's in the provided context."
        )
        
        self.knowledge_base = knowledge_base
        print("RAG Agent ready.")
    
    def set_knowledge_base(self, knowledge_base: KnowledgeBase):
        """
        Connect a knowledge base to this agent.
        
        Args:
            knowledge_base (KnowledgeBase): The knowledge base to use
        """
        self.knowledge_base = knowledge_base
        stats = knowledge_base.get_stats()  
        print(f"Knowledge base connected: {stats['collection_name']} ({stats['total_chunks']} chunks)")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant context from the knowledge base.
        
        Args:
            query (str): User's question
            top_k (int): How many chunks to retrieve
        
        Returns:
            list: Most relevant document chunks
        """
        if not self.knowledge_base:
            print("Warning: No knowledge base connected!")
            return []
        
        results = self.knowledge_base.query(query, top_k=top_k) 
        return results
    
    def build_prompt_with_context(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Build a prompt for the LLM using retrieved context.
        
        Args:
            query (str): User's question
            context_chunks (list): Retrieved document chunks
        
        Returns:
            str: Complete prompt for LLM
        """
        if not context_chunks:
            return f"""The user asked: "{query}"

You don't have any relevant information in your knowledge base to answer this question.
Please respond honestly that you don't have this information available, and suggest 
that the user might need to provide relevant documents or ask a different question."""
        
        context_text = "=== KNOWLEDGE BASE CONTEXT ===\n\n"
        context_text += "Here are relevant excerpts from the knowledge base:\n\n"
        
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['metadata'].get('source', 'Unknown Source')
            
            context_text += f"[Source {i}: {source}]\n"
            context_text += f"{chunk['text']}\n\n"
        
        prompt = f"""{context_text}
=== USER QUESTION ===

{query}

=== INSTRUCTIONS ===

Please answer the user's question using ONLY the information provided in the context above.

Important guidelines:
1. Cite which source(s) you used (e.g., "According to Source 1...", "Source 2 states...")
2. If the context contains the answer, provide it clearly and concisely
3. If the context doesn't fully answer the question, say so and explain what information is available
4. DO NOT make up information or use knowledge outside the provided context
5. Be helpful and conversational while staying factual

Your answer:"""
        
        return prompt
    
    def answer(self, query: str, top_k: int = 3, verbose: bool = True) -> Dict:
        """
        Answer a question using the full RAG pipeline.
        
        Args:
            query (str): User's question
            top_k (int): How many document chunks to retrieve
            verbose (bool): Print progress information
        
        Returns:
            dict: Contains answer, sources, confidence, and metadata
        """
        if verbose:
            print(f"Query: '{query}'")
        
        # Retrieval
        context_chunks = self.retrieve_context(query, top_k=top_k)
        
        if verbose and context_chunks:
            print(f"Found {len(context_chunks)} relevant chunks")
        elif verbose:
            print("No relevant context found")
        
        # Prompt Building
        prompt = self.build_prompt_with_context(query, context_chunks)
        
        # Generation
        answer = self.llm.generate(prompt)
        
        if verbose:
            print("Answer generated.")
        
        result = {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'text': chunk['text'][:300] + '...' if len(chunk['text']) > 300 else chunk['text'],
                    'metadata': chunk['metadata'],
                    'similarity': chunk.get('similarity', 0)
                }
                for chunk in context_chunks
            ],
            'num_sources': len(context_chunks),
            'has_sources': len(context_chunks) > 0
        }
        
        return result
    
    def interactive_mode(self):
        """
        Launch interactive Q&A mode.
        """
        print("\n=== INTERACTIVE RAG AGENT ===")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not question:
                    continue
                
                result = self.answer(question, verbose=False)
                
                print(f"\nAgent: {result['answer']}\n")
                
                if result['sources']:
                    print(f"Sources ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"   {i}. {source['metadata'].get('source', 'Unknown')}")
                print()
                
            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    print("\n=== RAG AGENT DEMO ===")
    
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            print("Please set GEMINI_API_KEY in your .env file!")
            sys.exit(1)
        
        # Initialize knowledge base
        kb = KnowledgeBase("gdg_rag_demo")
        
        # Add sample GDG documentation
        sample_docs = """
        GDG Event Registration and Participation Guide:
        
        Registration is free and open to all students. Visit gdg.community.dev to find 
        your local chapter and register for events. You'll need a Google account to sign up.
        
        Events typically run from 9:00 AM to 5:00 PM. We provide WiFi, power outlets, 
        coffee, snacks, and lunch. Please bring your laptop with a charger.
        
        Workshop Prerequisites:
        For our AI workshop, please ensure you have:
        - Python 3.8 or higher installed
        - A code editor (VS Code recommended)
        - 8GB RAM minimum
        - Enthusiasm to learn!
        
        What to Expect:
        Day 1 focuses on Python basics and NLP fundamentals.
        Day 2 covers vector databases and document processing.
        Day 3 is all about building RAG systems with Gemini AI.
        
        Certificates are provided to all participants who complete the workshop.
        """
        
        kb.add_document(
            sample_docs,
            metadata={
                'source': 'GDG Workshop Guide',
                'type': 'guidelines',
                'category': 'event-info'
            }
        )
        
        # Initialize RAG agent
        agent = RAGAgent(
            gemini_api_key=api_key,
            knowledge_base=kb,
            temperature=0.3
        )
        
        # Test queries
        test_questions = [
            "How much does it cost to attend GDG events?",
            "What should I bring to the workshop?",
            "What are the prerequisites for the AI workshop?",
            "Will I get a certificate?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nQuestion {i}: {question}")
            result = agent.answer(question, top_k=2, verbose=False)
            print(f"Answer: {result['answer']}\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}")