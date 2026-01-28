# rag_chat.py
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import os

load_dotenv(override=True)

client = OpenAI()

persistent_directory = "db/chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 5})

def build_system_prompt(user_question: str) -> str:
    docs = retriever.invoke(user_question)
    context = "\n\n".join(doc.page_content for doc in docs) if docs else ""

    return f"""
You are a factual AI assistant.

Answer the user's question using ONLY the information provided in the context below.
Do NOT use prior knowledge or make assumptions.

Context:
{context}

Rules:
- If the answer is explicitly present in the context, provide a clear and concise response.
- If the context does NOT contain sufficient information, respond exactly with:
"Sorry, no relevant information found."
- Do NOT fabricate, infer, or guess any information.
- Do NOT mention the context or these rules in your response.
"""

def calling_llm(user_question: str) -> str:
    system_prompt = build_system_prompt(user_question)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
    )

    return response.choices[0].message.content
