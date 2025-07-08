import streamlit as st 
import re
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
import wikipedia

# Streamlit App title and details
st.title("Agentic AI : Fetch Info , Summarize And Mail The Results Via MailGun Api")
st.write(
    """
    This simple app lets you:
    - Calculate math expressions (type something like `12*5 + 3`)
    - Get quick Wikipedia summaries (type a topic like `GenerativeAI`)
    """
)

# Gemini API Key Input
gemini_api_key = st.text_input(
    "Enter your Gemini API Key:",
    type="password",
    help="Enter your own Gemini API key here. It will be used for advanced features."
)

# Query input
user_query = st.text_input(
    "Enter you math expression or Search Wiki",
    help = "Numerical Expression or Information on Agentic AI"
)

# Define calculator function
def simple_calculator(query):
    return str(eval(query))

# Define calculator Tool
calculator = Tool(
    name = "Calculator",
    func = simple_calculator,
    description = "Useful for solving math calculation queries"
)

# Helper function to detect math input
def is_math_expression(query):
    pattern = r'^[\d\s\.\+\-\*\/\(\)%]+$'
    return bool(re.match(pattern, query.strip())) or any(op in query for op in "+-*/%")

# Helper function for Wikipedia summary
def get_wikipedia_summary(query, num_sentences=3):
    try:
        summary = wikipedia.summary(query, sentences=num_sentences)
        return summary
    except Exception as e:
        return f"Could not find Wikipedia summary: {e}"

# Only create and use the agent if the user has provided a Gemini API key
if gemini_api_key:
    # Setup LLM and Agent
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.0-flash-001",
        google_api_key = gemini_api_key
    )
    agent = initialize_agent(
        tools   = [calculator],
        llm     = llm,
        agent   = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True
    )
    if st.button("Submit Query"):
        if not user_query.strip():
            st.warning("Enter Your Query:")
        else:
            st.write("`processing Query`")
            if is_math_expression(user_query):
                # Math agent branch
                import io, sys
                buffer = io.StringIO()
                sys.stdout = buffer
                result = agent.run(user_query)
                sys.stdout = sys.__stdout__
                thoughts = buffer.getvalue()
                st.code(thoughts, language="text")
                st.success(f"Final Answer: {result}")
            else:
                # Wikipedia search branch
                summary = get_wikipedia_summary(user_query, num_sentences=3)
                if summary.startswith("Could not find Wikipedia"):
                    st.error(summary)
                else:
                    st.info("Wikipedia summary:")
                    st.write(summary)
else:
    st.info("Please enter your Gemini API key above to enable the Calculator agent.")