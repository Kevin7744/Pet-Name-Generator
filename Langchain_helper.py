from lanchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.Chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm = OpenAI(temperature = 0.7)
    # Prompt tempalate make it easy to generate prompts.
    prompt_template_name = PromptTemplate(
        # the prompt template are dynamic
        input_variables = ['animal_type', 'pet_color'],
        template = "I have a {animal_type} pet and it is {pet_color} in color. Suggest five cool names for my pet."
    )
    # Chain allows us to combine the 'llm' and prompt_template together.
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")
    response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})
    return response

# Create custom agents
def langchain_agent():
    llm=OpenAI(temperature=0.7)
    # pip install wikipedia library
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    ) 

    result = agent.run(
        "WHat is the average age of a dog? Multiply the age by 3"
    )
    print(result)

# print (generate_pet_name("cat"))
langchain_agent()