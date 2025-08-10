# llm_agent.py
import os
from groq import Groq

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"

ANSWER_PROMPT = """
You are a precise document-answering assistant.
Use ONLY the provided document clauses to answer the question.
DO NOT add information not found in the clauses.

Question: {question}

Context clauses (top relevant chunks):
{clauses}

Instructions:
- If the answer is present, extract and combine all relevant facts, conditions, and exclusions from the clauses.
- Include key eligibility requirements, waiting periods, limits, and exceptions exactly as stated.
- If multiple clauses contain relevant info, merge them into one coherent answer.
- If the answer is not in the clauses, return exactly: "Not stated in provided document."
- Cite all clause_ids used in square brackets, e.g., [doc.pdf_p3_c2, doc.pdf_p4_c1].
- Keep the answer concise (<= 120 words) but complete.

Return ONLY the answer text (no extra commentary).
"""


client = Groq(api_key=API_KEY)

def answer_question_with_context(question, topk_chunks):
    clauses_str = ""
    idsc = ""
    for c in topk_chunks:
        clauses_str += f"- [{c['chunk_id']}] (score={c['score']:.3f}) {c['text']}\n"
        idsc += f"{c['chunk_id']} score = {c['score']:.3f}\n"
    with open(f"clauses.txt", "w") as f:
            f.write(idsc)
    prompt = ANSWER_PROMPT.format(question=question, clauses=clauses_str)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        ans = resp.choices[0].message.content.strip()
        return ans
    except Exception as e:
        print(f"Error generating answer: {e}")
        if topk_chunks:
            print(f"Top chunk: {topk_chunks[0]['text']}")     # Print first 400 chars of top chunk for debugging
            return topk_chunks[0]["text"][:400]
        return "Error generating answer"
