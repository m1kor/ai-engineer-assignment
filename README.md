# Autify AI Engineer Assignment: Code Snippet Generation

This assignment evaluates your ability to write clean Python code and solve AI/LLM problems. Your main task is to implement the APIs necessary for generating code snippets.

The application should **at least** support the following programming languages:

- `Python`
- `Javascript`
- `Ruby`

For snippet description/feedback, the following languagues must be supported:

- `English`
- `Japanese`

## Reference Workflow

To understand this section, please open `design.html` in a browser. The design consists of two columns:

- **Left Column**: Contains a list of previously generated snippets
    - Click on a snippet to load it in the code editor
    - Option to delete a snippet
- **Right Column**: Contains the main code generator
    - Describe the code to generate and select the desired language in the text area
    - Clicking "Generate" will display the generated code snippet below
    - A feedback box is provided to improve the generated code
    - Button to generate tests for the code
    - Provide feedback to improve the generated tests
    - Button to run tests and display results
    - Disabled button to improve code based on test results

## Getting Started

To work on this project, you need a basic understanding of FastAPI, Python, and Javascript. Implement the backend APIs using FastAPI. For frontend work, you have two options:

- Implement frontend logic by dividing the design page into multiple parts and implementing logic in the backend using FastAPI template(s).
- Write Javascript code to create a single-page app and interact with the backend using APIs. If not familiar with frontend work, use ChatGPT or another AI tool to assist.

The repository's basic settings are configured to run in a specific way. A Dockerfile installs requirements and runs `app.py`, so refrain from updating the app structure unless necessary.

Use OpenAI ChatGPT 3.5 turbo model for this project. Sign up on ChatGPT to obtain the API key. Development and testing should cost less than $0.1.

To start working on the project:
1. You will be provided with a zip file containing the project structure.
2. Set up a new Python 3.11 environment and install dependencies.
3. Create the .env file (see .env.example) and add the required environment variables. Update .env.example with correct variable names.
4. Understand the application logic and frontend workings by reviewing `design.html`.
5. Work on `design.html` and `app.py` to create a coherent system meeting assignment requirements. Create additional files if necessary.

Before submission, ensure the project works locally. Run `start-docker-server.sh` to start the server and visit `http://localhost:8000` to see the project. Docker must be installed.

Once completed, send the repository link to the HR team for evaluation.

> You're encouraged to use any AI tool like ChatGPT or Copilot to assist in code writing. Spend no more than 4 hours on this project.

## Evaluation

For evaluation, your code must run without errors. If `start-docker-server.sh` throws errors due to your code during image building or server running, you'll be disqualified.

Your code should meet all assignment requirements. Marks will be deducted for any unmet requirement:

- Code Quality (15%)
    - Your code should be clean and easy to read
    - Your code should use black code formatter
    - Your code should be well documented
    - Your code should be efficient
    - Github Repository should be clean and well organized including commit messages
- UI/UX (15%):
    - The UI is faithful to the provided design template
    - The code blocks only contains the code snippet, no explanation or other text
    - Sections are added when needed i.e. test generation part is hidden until we generate the code
    - Language Highlighter works proper for all of the supported languages
    - Selecting a code snippet hides `Delete` button and highlights the selected snippet
    - The code is streamed to the UI as it is generated
- Functionality (70%)
    - Snippet List (10%):
        - We should be able to click and load previously generated snippets
        - We should be able to delete previously generated snippets
        - We should be able to create new snippets
        - Snippet should automatically be saved when we click the "Generate" button
        - The title & language of the snippets are generated automatically
    - Code Generation (50%):
        - We should be able to generate code snippets (functions) in Python, Javascript, and Ruby
        - We should be able to provide feedback in English or Japanese on the generated code and improve it
        - We should be able to generate tests for the code snippets (non-function code)
        - We should be able to improve the tests with feedback in English or Japanese
        - We should be able to run the tests to validate the code (Only Enabled when Languague Python)
        - We should be able to improve the code based on test results when tests fail
    - Miscellanous (10%):
        - LLM Security: Prompt injection should be handled properly
        - High quality code generation using techniques like Chain-of-thought, etc.
