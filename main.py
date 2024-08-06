from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from langchain_community.llms.ollama import Ollama
from crewai_tools import SerperDevTool
import os
import json

load_dotenv()
os.environ["OPENAI_API_KEY"] = "NA"
api_key = os.environ.get('SERPER_API_KEY')
country = input("Which country: ")
city = input("Which city: ")

search_engine = SerperDevTool(
    search_url=f"""https://google.serper.dev/search?api_key=${api_key}""",
    n_results=2,
)

llm = Ollama(
    model="llama3.1:8b-instruct-q8_0",
    base_url="http://localhost:11434")

general_agent = Agent(role="Weather Anchor",
                      goal=f"""Give forecast of the weather in the city of {city}.""",
                      backstory=f"""You have been working in one of the main TV channels of {country} for 20 years and 
                      you are the leading Weather forecast anchor.""",
                      allow_delegation=False,
                      verbose=True,
                      max_iter=1,
                      llm=llm)

task = Task(description=f"""What is the weather in the city {city} of the country {country}""",
            agent=general_agent,
            expected_output="A simple weather forecast, saying temperature in degrees Celsius, "
                            "humidity and forecasts for the rest of the day. "
                            "Use the most resent data available",
            tools=[search_engine])

crew = Crew(
    agents=[general_agent],
    tasks=[task],
    verbose=2
)

crew_output = crew.kickoff()

print(f"Raw Output: {crew_output.raw}")
if crew_output.json_dict:
    print(f"JSON Output: {json.dumps(crew_output.json_dict, indent=2)}")
if crew_output.pydantic:
    print(f"Pydantic Output: {crew_output.pydantic}")
print(f"Tasks Output: {crew_output.tasks_output}")
print(f"Token Usage: {crew_output.token_usage}")
