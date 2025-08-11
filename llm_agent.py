from groq import Groq
import logging
from typing import List, Dict, Any
from config import Config

logger = logging.getLogger(__name__)

class LLMService:
    """Memory-optimized LLM service"""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Groq client"""
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        try:
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            logger.info("Groq client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer with context, optimized for memory"""
        if not context_chunks:
            return "Not stated in provided document."
        
        try:
            # Build context string efficiently
            context_parts = []
            total_length = 0
            max_context_length = 300000  # Limit context to manage memory
            
            for chunk in context_chunks:
                chunk_text = f"- [{chunk['chunk_id']}] (score={chunk['score']:.3f}) {chunk['text']}"
                if total_length + len(chunk_text) > max_context_length:
                    break
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            
            context_str = "\n".join(context_parts)
            
            # Create prompt
            prompt = self._create_prompt(question, context_str)
            
            # Generate response
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.MAX_ANSWER_LENGTH + 50,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Log chunk IDs for debugging
            chunk_ids = [chunk['chunk_id'] for chunk in context_chunks[:3]]
            logger.info(f"Generated answer using chunks: {chunk_ids}")
            
            print(answer, " The answeris here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fallback to first chunk if available
            if context_chunks:
                return context_chunks[0]["text"][:400] + "..."
            return "Error generating answer"
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create optimized prompt"""
        return f"""You are a precise document-answering assistant.
Use ONLY the provided document clauses to answer the question.
DO NOT add information not found in the clauses.

Question: {question}

Context clauses (top relevant chunks):
{context}

Instructions:
- If the answer is present, extract and combine all relevant facts, conditions, and exclusions.
- Include key eligibility requirements, waiting periods, limits, and exceptions exactly as stated.
- If multiple clauses contain relevant info, merge them into one coherent answer.
- If the answer is not in the clauses, return exactly: "Not stated in provided document."
- Cite all clause_ids used in square brackets, e.g., [doc.pdf_p3_c2].
- Keep the answer concise (<= {Config.MAX_ANSWER_LENGTH} words) but complete.

Return ONLY the answer text (no extra commentary)."""

# Global LLM service instance
llm_service = LLMService()

# =============================================================================

