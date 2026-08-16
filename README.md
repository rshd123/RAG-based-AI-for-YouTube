# YouTube Transcript RAG

This project lets you ask a language model about the content of a YouTube video. It uses the video's transcript as the source of information, so the model is instructed to answer only from the retrieved transcript sections.

## How the system works

1. The application asks for a YouTube video ID.
2. It fetches the video's available transcript using `youtube-transcript-api`.
3. The transcript is split into overlapping text chunks.
4. Each chunk is converted into an embedding using the `sentence-transformers/all-MiniLM-L6-v2` Hugging Face model.
5. The embeddings are stored in an in-memory FAISS vector database.
6. The application retrieves the four chunks most relevant to the question.
7. The retrieved chunks and the question are sent to Groq's `llama-3.1-8b-instant` model.
8. The model returns an answer based only on the supplied transcript context.

The current question is configured in `main.py` as:

```python
QUESTION = "Can you summarize the video?"
```


## Requirements

- Python 3.10 or newer
- A Groq API key
- Internet access to fetch YouTube transcripts and download the embedding model on its first use

Install the required packages in your active Python or Conda environment:

```powershell
pip install groq python-dotenv youtube-transcript-api langchain-text-splitters langchain-community langchain-core langchain-groq sentence-transformers faiss-cpu
```

## Configure the API key

Create a `.env` file in the project folder:

```env
GROQ=your_groq_api_key
```

Do not commit or share this file, because it contains your private API key.

## Run the application

Activate your Conda environment if you use one, then run:

```powershell
python main.py
```

When prompted, enter a YouTube video ID, for example:

```text
Enter the YouTube video ID: rjBsQ9SygnE
```

Use only the ID portion of a YouTube link. For example, for `https://www.youtube.com/watch?v=rjBsQ9SygnE`, the video ID is `rjBsQ9SygnE`.

## Error handling

The application reports clear errors when:

- The `GROQ` value is missing from `.env`.
- No video ID is entered.
- The video has no transcript or transcripts are disabled.
- The transcript is empty.
- The embeddings or FAISS vector database cannot be created.
- The Groq model cannot be initialized.

## Language behavior

The current transcript request does not specify a language:

```python
YouTubeTranscriptApi().fetch(video_id)
```

As a result, it defaults to English captions. To support another language, the code should pass the preferred language code, such as `languages=["hi"]` for Hindi, when fetching the transcript. The requested caption language must be available for the selected YouTube video.

<!-- ## Current limitations

- The video must have an accessible YouTube transcript.
<!-- - The vector database exists only while the program is running; it is not saved to disk. -->
<!-- - The question is currently fixed to a request for a video summary. -->
<!-- - The application accepts a video ID, not a complete YouTube URL. --> -->
