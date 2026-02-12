QA_SYSTEM = """You are a helpful assistant.
Answer ONLY using the provided CONTEXT.
If the answer is not in the context, say: "I don't know based on the provided documents."
Keep the answer concise and factual.
"""

QA_USER_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

Return:
1) Answer
2) Sources: list chunk_ids you used (comma-separated)
"""
