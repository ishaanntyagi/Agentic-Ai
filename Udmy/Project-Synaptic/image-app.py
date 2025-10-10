import os
import base64
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


load_dotenv()
print(".env successfully loaded")


import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key    = os.getenv("OPENAI_API_KEY")
gemini_api_key    = os.getenv("GEMINI_API_KEY")
groq_api_key      = os.getenv("GROQ_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
print("groq Key loaded")


llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct")
print(f"The model loaded is : {llm}")

image_path = r"C:\Users\ishaa\OneDrive\Desktop\Projects\Medical_Image_Captioning\Testing_SampleImages\G\Te-gl_0072.jpg"
print(f"image loaded sucessfully {image_path}")


# Read the image file in binary mode and encode it in base64
with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")


image_type = image_path.split('.')[-1]
image_data_uri = f"data:image/{image_type};base64,{image_base64}"




message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What is in this image? Describe it in detail also What is this kind of Tumor , Explain In detail? ",
        },
        {
            "type": "image_url",
            "image_url": {"url": image_data_uri},
        },
    ]
)


api_response = llm.invoke([message])


print(" \n \n \n \n ----------------------------------- HealthChatBot's Response --------------------------------------------")
print(api_response.content)
print("---------------------------------------------------------------------------------------------------------")
