"""Ask questions about a YouTube video's transcript using a local FAISS index."""

import os
import sys

from dotenv import load_dotenv #type:ignore
from groq import Groq #type:ignore
from langchain_community.embeddings import HuggingFaceEmbeddings #type:ignore
from langchain_community.vectorstores import FAISS #type:ignore
from langchain_core.output_parsers import StrOutputParser #type:ignore
from langchain_core.prompts import PromptTemplate #type:ignore
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough #type:ignore
from langchain_groq import ChatGroq #type:ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter #type:ignore
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi #type:ignore


QUESTION = "Can you summarize the video?"


def format_docs(retrieved_docs):
    """Combine retrieved transcript chunks into the prompt context."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def fetch_transcript(video_id: str) -> str:
    """Fetch and normalize a video's transcript."""
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        text = " ".join(item.text for item in transcript)
    except TranscriptsDisabled:
        raise RuntimeError("This video has transcripts disabled.") from None
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch a transcript for video ID '{video_id}'. "
            "Check that the ID is valid and that a transcript is available."
        ) from exc

    if not text.strip():
        raise RuntimeError("The video transcript is empty.")

    return text


def build_retriever(transcript_text: str):
    """Split, embed, and index the transcript in an in-memory FAISS store."""
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript_text])
        if not chunks:
            raise RuntimeError("No chunks could be created from the transcript.")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Could not create the vector database.") from exc


def create_chain(api_key: str, retriever):
    """Create the notebook's LangChain RAG pipeline."""
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}

Question: {question}
""",
        input_variables=["context", "question"],
    )

    try:
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.2,
            groq_api_key=api_key,
        )
        parallel_chain = RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        return parallel_chain | prompt | llm | StrOutputParser()
    except Exception as exc:
        raise RuntimeError("Could not initialize the Groq language model.") from exc


def main():
    load_dotenv()
    api_key = os.getenv("GROQ")
    if not api_key:
        print("Error: GROQ is missing. Add GROQ=your_api_key to your .env file.")
        return 1

    # Keeps the same initial API-client setup as the notebook.
    try:
        Groq(api_key=api_key)
    except Exception as exc:
        print(f"Error: Could not configure the Groq client: {exc}")
        return 1

    try:
        video_id = input("Enter the YouTube video ID: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo video ID was provided.")
        return 1

    if not video_id:
        print("Error: A YouTube video ID is required.")
        return 1

    try:
        transcript_text = fetch_transcript(video_id)
        retriever = build_retriever(transcript_text)
        chain = create_chain(api_key, retriever)
        answer = chain.invoke(QUESTION)
        print(answer)
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
