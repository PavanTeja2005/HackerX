# llm_agent.py
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")  # or use any other provider

ANSWER_PROMPT = """
You are a precise document-answering assistant. Use ONLY the provided document clauses to answer the question.
Question: {question}

Context clauses (top relevant chunks):
{clauses}

Task: Provide a concise, factual answer (<= 120 words). If the answer is present in the clauses, cite clause_ids in square brackets, e.g., [doc.pdf_p3_c2]. If not found, return "Not stated in provided document."

Return ONLY the answer text (no extra metadata).
"""

def answer_question_with_context(question, topk_chunks):
    clauses_str = ""
    for c in topk_chunks:
        clauses_str += f"- [{c['chunk_id']}] (score={c['score']:.3f}) {c['text']}\n"
    prompt = ANSWER_PROMPT.format(question=question, clauses=clauses_str)
    # Call OpenAI completion (ChatCompletion recommended)
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or gpt-4 if allowed; fall back to gpt-4o-mini for speed/cost
            messages=[{"role":"system","content":"You are a helpful assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.0
        )
        ans = resp["choices"][0]["message"]["content"].strip()
        return ans
    except Exception as e:
        # Fallback: simple extractive heuristic
        if topk_chunks:
            return topk_chunks[0]["text"][:400]
        return "Error generating answer"
