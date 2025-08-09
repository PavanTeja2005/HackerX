# llm_agent.py
import os
from groq import Groq

API_KEY = "groq-api"
MODEL = "llama-3-3-70b-versatile"

ANSWER_PROMPT = """
You are a precise document-answering assistant. Use ONLY the provided document clauses to answer the question.
Question: {question}

Context clauses (top relevant chunks):
{clauses}

Task: Provide a concise, factual answer (<= 120 words). If the answer is present in the clauses, cite clause_ids in square brackets, e.g., [doc.pdf_p3_c2]. If not found, return "Not stated in provided document."

Return ONLY the answer text (no extra metadata).
"""

client = Groq(api_key=API_KEY)

def answer_question_with_context(question, topk_chunks):
    clauses_str = ""
    for c in topk_chunks:
        clauses_str += f"- [{c['chunk_id']}] (score={c['score']:.3f}) {c['text']}\n"
    prompt = ANSWER_PROMPT.format(question=question, clauses=clauses_str)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.0
        )
        ans = resp.choices[0].message.content.strip()
        return ans
    except Exception as e:
        if topk_chunks:
            return topk_chunks[0]["text"][:400]
        return "Error generating answer"
