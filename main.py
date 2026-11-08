
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough
)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import re
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

loader = TextLoader(
    os.path.join(BASE_DIR, "plants.txt"),
    encoding="utf-8"
)

loaded_documents = loader.load()

documents = [
    Document(
        page_content=doc.page_content,
        metadata=doc.metadata
    )
    for doc in loaded_documents
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

chunks = text_splitter.split_documents(documents)

context = "\n\n".join(
    chunk.page_content for chunk in chunks
)

prompt = ChatPromptTemplate([
    (
        "system",
        """
        Ты AI Plant Doctor — помощник по уходу за комнатными растениями.

        Строгие правила:
        1. Отвечай исключительно на русском языке.
        2. Запрещено использовать английские слова.
        3. Используй только информацию из базы знаний.
        4. Не добавляй информацию от себя.
        5. Не повторяй одну и ту же мысль.
        6. Отвечай коротко, максимум 2 предложения.
        7. Не используй приветствия и заключения.
        8. Не используй списки и Markdown.
        9. Если информации нет, отвечай: Я не знаю.

        Информация о растениях:
        {context}
        """
    ),
    (
        "human",
        "{text}"
    )
])

model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)

prepare_data = RunnableParallel({
    "context": RunnableLambda(lambda x: context),
    "text": RunnablePassthrough()
})

chain = prepare_data | prompt | model | StrOutputParser()

test_app = FastAPI(
    title="AI Plant Doctor",
    description="AI помощник по уходу за комнатными растениями",
    version="1.0"
)


class QuestionSchema(BaseModel):
    question: str


@test_app.get("/")
async def home():
    return {
        "project": "AI Plant Doctor",
        "status": "AI is working",
        "message": "Задайте вопрос о комнатных растениях"
    }


@test_app.post("/questions/")
async def questions_for_answer(data: QuestionSchema):

    question = data.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Введите вопрос"
        )

    try:
        answer = chain.invoke(question)

        replacements = {
            "plenty": "достаточно",
            "completely": "полностью",
            "beautiful": "красивым",
            "maintain": "поддерживать"
        }

        for word, replacement in replacements.items():
            answer = re.sub(
                rf"\b{word}\b",
                replacement,
                answer,
                flags=re.IGNORECASE
            )

        answer = re.sub(r"\b[A-Za-z]+\b", "", answer)
        answer = re.sub(r"\n+", " ", answer)
        answer = re.sub(r"\s{2,}", " ", answer).strip()

        return {
            "question": question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    uvicorn.run(
        test_app,
        host="127.0.0.1",
        port=8000
    )
