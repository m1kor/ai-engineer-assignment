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
async def stream(id: str):
    # Remove prompts older than the configured timeout
    for key in list(prompts.keys()):
        if time.time() - prompts[key]["time"] > timeout:
            del prompts[key]

    # SSE generator
    async def stream_response(messages):
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
        messages = [
            {
                "role": "system",
                "content": prompts[id]["system"],
            },
            {"role": "user", "content": prompts[id]["input"]},
        ]
        del prompts[id]
        return EventSourceResponse(stream_response(messages))
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
