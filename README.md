# Agentic-Ai


**Agentic-Ai** is a beginner-friendly project to help you learn and experiment with Generative AI “agents” using Python. This repository offers super-simple code and step-by-step explanations, so anyone—even with no AI background—can start building agentic systems that combine language models, tools, and external APIs.

---

## 🖼️ App Demo Screenshots

Below are actual outputs from the Streamlit app:

### Wikipedia Summarization
![Screenshot 1](Agent-Project-pipeline/Outputs/Screenshot%202025-07-08%20150623.png)

### Math Calculation
![Screenshot 2](Agent-Project-pipeline/Outputs/Screenshot%202025-07-08%20150938.png)

### Error Handling
![Screenshot 3](Agent-Project-pipeline/Outputs/Screenshot%202025-07-08%20151206.png)

---

## 🧑‍🎓 What is a Generative AI Agent?

A **Generative AI agent** is a program that uses a large language model (LLM) (like Gemini or GPT) and connects it to tools (such as a calculator, API, or web search) to solve tasks. The agent “decides” what steps to take, which tools to use, and in what order, to answer your query.

**Example:**  
If you ask, “What’s the capital of France squared?”  
- The agent fetches “Paris” (from Wikipedia), realizes it needs to perform an operation, and can use a calculator tool if needed.

---

## 🏗️ What’s in this Repository?

- **Basic Agent Notebook** (`01-Agents-Basic.ipynb`):  
  Shows how to build a simple agent with LangChain and Gemini that solves math expressions using a custom calculator tool.

- **Weather API Notebook** (`02-Weather-Api.ipynb`):  
  Demonstrates how to call an external weather API and let the agent fetch real-world data.

- **Wikipedia Agent Notebook** (`03- wikipedia-Agent.ipynb`):  
  Teaches how to integrate Wikipedia search and summarization into your agent.

- **Pipeline Streamlit App** (`Agent-Project-pipeline/App.py`):  
  An easy-to-use web app where you can type questions and see the agent in action—doing calculations or Wikipedia searches using Gemini as the LLM.

---

## 🧩 How Does the Agentic-Ai Wiki + Calculator Agent Work?

The **Agentic-Ai Agent** combines:
- A **Large Language Model (LLM)**: Gemini, accessed via the Gemini API.
- **Tools**:
  - A **Calculator**: For math expressions.
  - **Wikipedia Search**: For knowledge queries.
