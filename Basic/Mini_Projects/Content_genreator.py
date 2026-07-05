from dotenv import load_dotenv #api load in this file
from langchain_google_genai import ChatGoogleGenerativeAI ## model instailzation function for google 
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel ## uses form thre class to defince the a how to ai will send the response 
from langchain_core.output_parsers import  PydanticOutputParser
from typing import List , Optional 
#load api 
load_dotenv()

#model instialization
model =ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.8,
)

#base class for structured ouptut
class Content_Generator(BaseModel):
    Short_Description:str
    Long_Description:str
    Relevant_keywords:List[str]
    SEO_Tags:str
    Video_Category:List[str]
    Suggested_Thumbnail_Text:Optional[str]
    Call_To_Action:str
    Alternative_Titles:List[str]
    Tips_to_Improve_Discoverability:str

#it is the parse that will parse these struct ouptut lay out to model
parser= PydanticOutputParser(pydantic_object=Content_Generator)

#this is the alternative of humman,ai,system message used in pervoius case with better approach here we can define the role of model to be act as who?
prompt= ChatPromptTemplate.from_messages(
    [
           ('system',"""

You are an expert Content Creation and SEO Assistant specializing in content for creators, influencers, and social media professionals.

Your responsibility is to create engaging, SEO-optimized content based on the information provided by the user.

Your objectives are:
- Write content that is clear, engaging, and optimized for search engines.
- Generate descriptions that naturally include relevant keywords.
- Produce content appropriate for the selected platform.
- Maintain the requested tone throughout the content.
- Create compelling calls to action that encourage engagement.
- Suggest alternative titles that are more clickable while remaining relevant.
- Generate SEO tags and keywords that improve discoverability.
- Suggest a short thumbnail text that is catchy and easy to read.
- Provide practical tips to improve discoverability on the selected platform.

Important Rules:
- Do not invent information that is not provided.
- Keep descriptions natural and avoid keyword stuffing.
- Make the content audience-focused.
- Ensure all generated content is unique.
- Follow the formatting instructions exactly.

{format_instruction}

"""),
("human", """
Generate content using the following information.

Title:
{title}

Platform:
{platform}

Content Category:
{category}

Target Audience:
{audience}

Tone:
{tone}

Primary Language:
{language}

""")
    ]
)
#here we are get the information from user 
if __name__ == "__main__":

    print("-----------------Enter your Detail------------------------")

    title = input("Enter the title: ")
    platform = input("Enter the platform: ")
    category = input("Enter the category: ")
    audience = input("Enter the audience: ")
    tone = input("Enter the tone: ")
    language = input("Enter the language: ")

    final_prompt = prompt.invoke(
        {
            "title": title,
            "platform": platform,
            "category": category,
            "audience": audience,
            "tone": tone,
            "language": language,
            "format_instruction": parser.get_format_instructions(),
        }
    )

    response = model.invoke(final_prompt)

    content_data = parser.parse(response.content[0]["text"])

    print(content_data)