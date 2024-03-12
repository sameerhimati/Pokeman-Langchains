from astrapy.db import AstraDB
from astrapy.ops import AstraDBOps
from dotenv import load_dotenv
from langchain.memory import AstraDBChatMessageHistory, ConversationBufferMemory
from langchain.llms import OpenAI
from langchain import LLMChain, PromptTemplate
import os

load_dotenv()

astra_token = os.getenv("astra")
openai_api_key = os.getenv("OPENAI_API_KEY")

#client = AstraDBOps(token=astra_token)

# Initialize the client
db = AstraDB(
  token=astra_token,
  api_endpoint="https://34f4c19a-d4a4-4810-83cc-d7fa63ffaa42-us-east1.apps.astra.datastax.com")

print(f"Connected to Astra DB: {db.get_collections()}")


message_history = AstraDBChatMessageHistory(
    session_id = "savefile", 
    api_endpoint='https://34f4c19a-d4a4-4810-83cc-d7fa63ffaa42-us-east1.apps.astra.datastax.com',
    token=astra_token
)

message_history.clear()

astra_buff_memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history
)

template = """
"You are now the guide of an adventure game based on the idea of Pokemon.
The difference is its a completely new region called Mastion with various different paths to the adventures ending.
Its your job to guide the trainer across Mastion in the typical fashion of a Pokemon game.
Mastion is a region with no legendaries and pokemon from all the regions of the Pokemon games so far.
The progression is the same as a typical pokemon game, however, you need to make up cities, challenges, puzzles and other tasks the trainer will encounter during the journey.
You must navigate them through the journey, dynamically adapting the tale based on the traveler's decisions.
At all times you must maintain a list of the following: [Pokemon the trainer has with their levels and the 4 moves that pokemon knows, a map of all the landmarks and cities they visited, the number of gym badges, the amount of money they have].
The trainer is given a choice of 3 Pokemon to start the adventure, these are typically weaker pokemon, start at Level 5 and have a total of 3 stages of evolution, typically its between Fire, Water and Grass and are of equal skill level.
The trainer has to beat 8 gyms, collect their badges and then beat the elite 4 and the champion. 

Your goal is to create this branching narrative experience where each choice leads to a new path, ultimately determining the trainer's fate.

The rest you adapt based on your knowledge of Pokemon and Pokemon games. 

Here are som rules to follow:
0. Do not make up answers for the human, instead ask the trainer for them. After every question you ask wait for a response. 
1. Start by asking the trainer's name. After they give it wait to confirm their name, ask them again.
2. Ask if they have ever played a pokemon like game, if not, give them a rundown of the objectives and rules.
3. Before proceeding, ask them if they undersood the objective. Then, Give the player a choice of the 3 starter pokemon of your choosing.
4. Have paths that lead to success for the trainer. 
5. Have some paths to failure but the ultimate goal is that the trainer wins, if the trainer fails at any point, explain why they fail and end with the text: "The End.", I will search for this text to end the game.
6. At any point in the game if the user types "Exit" then end the game and stop generating responses, I will search for this text to end the game.

Here is the chat history, use this to understand what to say next: {chat_history}

Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables = ["chat_history", "human_input"],
    template = template
)

llm = OpenAI(openai_api_key=openai_api_key)
llm_chain = LLMChain(
    llm=llm,
    prompt = prompt,
    memory = astra_buff_memory
)
choice = input("Your reply: ")
while True:
    response = llm_chain.predict(human_input=choice)
    print(response.strip())

    if "The End." in response:
        break

    choice = input("Your reply: ")


message_history.clear()