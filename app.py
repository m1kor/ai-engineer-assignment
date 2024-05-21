import json
import multiprocessing
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from openai import AsyncOpenAI
from uuid import uuid4

app = FastAPI()
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
snippets = {}  # All client snippets within timeout


# Code snippet model
class CodeSnippet(BaseModel):
    description: str
    source: str
    title: str
    language: str


# Incoming prompt model
class IncomingPrompt(BaseModel):
    system: str
    input: str


# Incoming source code model
class IncomingCode(BaseModel):
    source: str
    language: str


prompts = {}  # All client prompts within timeout
model = "gpt-3.5-turbo"  # Model to use for the LLM
timeout = 60  # Prompt storage timeout


# Root route
@app.get("/")
def root():
    return FileResponse("design.html")


# Register a new prompt
@app.post("/prompt")
async def prompt(message: IncomingPrompt):
    id = str(uuid4())  # Generate a unique ID
    prompts[id] = {
        "system": message.system,
        "input": message.input,
        "time": time.time(),
    }
    return {"id": id}


# Stream LLM completions for a prompt
@app.get("/stream")
async def stream(id: str, generate_title: bool = False):
    # Remove prompts older than the configured timeout
    for key in list(prompts.keys()):
        if time.time() - prompts[key]["time"] > timeout:
            del prompts[key]

    # SSE generator
    async def stream_response(messages, title=None):
        def preprocess(
            content: str,
        ) -> str:  # Preprocess the content before sending it to the client
            if "```" in content:
                # Return everything only inside code blocks
                lines = "\n".join(
                    [
                        "\n```" if line.strip()[:3] == "```" else line
                        for line in content.split("\n")
                    ]
                ).split(
                    "\n```"
                )  # Normalize code blocks (remove specified language and place code block tags on separate lines)
                return "\n".join(
                    [lines[i] for i in range(1, len(lines), 2)]
                ).strip()  # Return only code block contents
            else:
                return content

        # Yield title
        if title is not None:
            yield {
                "event": "title",
                "data": json.dumps(
                    {"title": title},
                ),
            }

        stream = await openai.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        content = ""
        language = None
        async for chunk in stream:
            content += chunk.choices[0].delta.content or ""
            if language is None:
                lines = content.split("\n")[:-1]
                for line in lines:
                    if line.strip()[:3] == "```":
                        parts = line.split("```")
                        if len(parts) > 1:
                            language = parts[1].strip()
                            yield {
                                "event": "language",
                                "data": json.dumps(
                                    {"language": language or "plaintext"}
                                ),
                            }
                            break
            yield {
                "event": "append",
                "data": json.dumps(
                    {
                        "content": preprocess(content),
                    }
                ),
                "retry": 500,
            }
        # Get the programming language from the first code block
        lines = content.split("\n")
        yield {"event": "end", "data": ""}

    if id in prompts:
        system_prompt = prompts[id]["system"]
        user_prompt = prompts[id]["input"]
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ]
        del prompts[id]
        # Generate title
        title = None
        if generate_title:
            title_messages = [
                {
                    "role": "system",
                    "content": "Summarize user input by providing as short title as possible (5 words max) while preserving the initial meaning for the following user prompt. Omit the programming language name:",
                },
                {"role": "user", "content": user_prompt},
            ]
            title_completion = await openai.chat.completions.create(
                model=model,
                messages=title_messages,
            )
            title = (
                title_completion.choices[0].message.content.strip().rstrip(".")
            )  # Remove trailing period
        return EventSourceResponse(stream_response(messages, title))
    else:
        raise HTTPException(status_code=404, detail="Entry not found")


# Execute Python code
@app.post("/execute")
async def execute(code: IncomingCode):
    if code.language is None or code.language.strip().lower() != "python":
        raise HTTPException(status_code=400, detail="Only Python code is supported")
    else:

        def execute_code(source):
            exec(source)

        # Create a new process
        process = multiprocessing.Process(target=execute_code, args=(code.source,))
        # Start the process
        process.start()
        # Wait for the process to finish
        process.join()

        # Return the exit code of the process
        return {"exit_code": process.exitcode}


# REST for Code Snippets


# Get all code snippets
@app.get("/snippets")
async def get_snippets():
    return {
        key: {"title": value["title"], "language": value["language"]}
        for key, value in snippets.items()
    }


# Create a new code snippet
@app.post("/snippet")
async def post_snippet(snippet: CodeSnippet):
    id = 0
    while id in snippets:
        id += 1
    snippets[id] = {
        "description": snippet.description,
        "source": snippet.source,
        "title": snippet.title,
        "language": snippet.language.capitalize(),
    }
    return {"id": id}


# Get a code snippet
@app.get("/snippet/{id}")
async def get_snippet(id: int):
    if id in snippets:
        return snippets[id]
    else:
        raise HTTPException(status_code=404, detail="Entry not found")


# Delete a code snippet
@app.delete("/snippet/{id}")
async def delete_snippet(id: int):
    if id in snippets:
        del snippets[id]
        return {"status": "OK"}
    else:
        raise HTTPException(status_code=404, detail="Entry not found")
