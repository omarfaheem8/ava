import random

import gradio as gr

import requests

from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper

from langchain.chat_models import ChatOpenAI

import os

import constants

 

os.environ["OPENAI_API_KEY"] = constants.APIKEY

 

# Initialize the index variable

index = None

 

# Function to fetch user balance from an API

def fetch_balance(user_id):

    api_url = "http://175.107.192.244:8082/API/User/AvailableBalance"

    headers = {"Content-Type": "application/json"}

    payload = {

        "codId": user_id

    }

 

    response = requests.post(api_url, json=payload, headers=headers)

 

    if response.status_code == 200:

        balance_data = response.json()

        available_balance = balance_data.get("availableBalance", "Balance data not available")

        return available_balance

    else:

        return "Error fetching balance"

 

# Construct the index

def construct_index(directory_path):

    max_input_size = 4096

    num_outputs = 512

    max_chunk_overlap = 20

    chunk_size_limit = 600

 

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

 

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

 

    documents = SimpleDirectoryReader(directory_path).load_data()

 

    global index  # Use the global index variable

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

 

    index.save_to_disk('index.json')

 

    return index

 

# Chatbot function

def chatbot(message, history):

    input_text = message

    if "balance" in input_text.lower() or "available balance" in input_text.lower():

        balance = fetch_balance("48421")

        response = f"Your balance is: {balance}"

    else:

        response = index.query(input_text, response_mode="compact").response

 

    return response

 

# HTML code for logo image

logo_html = f'<div style="display: inline-block; vertical-align: middle;"><img src="https://covglobal.blob.core.windows.net/img/aiMulti.png" style="max-width: 450px; margin-right: 10px;" /></div>'

 

# CSS for footer

css = """

gradio-app{
background-image: url("https://covglobal.blob.core.windows.net/img/ai.jpg") !important;
font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
  margin: 0;
  padding: 0;
  background-repeat: no-repeat !important;
  background-size: cover !important;
}

h1{
color: white !important;
font-size:30px !important;
font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
}

.svelte-nab2ao{
background-color: #212121 !important;
}

#component-2{
width:54% !important;

margin-left:46% !important;

}

#component-10{
display:none;
}

#component-11{
display:none;
}

#component-3{
box-shadow: 0px 0px 10px 0px #0F1211, 0px 0px 10px 0px #C24FFF !important;
}

.svelte-90oupt{

}
.bot{
background-color: #7AC8F5 !important;
}

title{
font-size:30px !important;
}

body {

    background-color: black;
    color: white; /* Set text color to white for visibility */
    background-image: url('https://d1hbpr09pwz0sk.cloudfront.net/logo_url/covalent-private-eddbaca3'); /* Replace YOUR_IMAGE_URL with the actual URL of your background image */
    background-size: cover; /* Cover the entire background */
    background-repeat: no-repeat; /* Do not repeat the background */
    background-attachment: fixed; /* Fixed background position */

}

 .gr-header {
   
    color: #ffffff !important; 
    padding: 10px !important; 
    font-size: 24px !important;
}

footer {

    visibility: hidden;

    position: relative;

}

 footer::before {

    visibility: visible;

    content: "";

    background-image: url('https://covalent.global/wp-content/uploads/2023/03/logo.png');

    background-repeat: no-repeat;

    width: 20%;

    height: 145px;

    position: absolute;

    bottom: -300%;

    left: 50%;

    transform: translateX(-50%);

    display: flex;

    flex-direction: column;

    align-items: center;

   

}

 

 

 

footer::before span {

    font-size: 16px; /* Adjust the font size */

    font-weight: bold; /* Make it bold */

    display: block; /* To ensure line breaks */
}

 

@media only screen   
and (min-width: 2000px)  
and (max-width: 3005px)  
{ 
#component-3{
height:600px !important;
}

#component-2{
width:55% !important;

}

}  

"""

 

 

# Create the Gradio ChatInterface

demo = gr.ChatInterface(chatbot, css=css, title=logo_html)

 

if __name__ == "__main__":

    # Construct the index before launching the interface

    construct_index("docs")

    demo.launch(share=True)