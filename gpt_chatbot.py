# this is essentially the same as amanda_chatbot.py but uses the openai API for chatgpt
import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,  LLMChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(page_title="Amanda - An LLM-powered Streamlit app for therapy")


api_key = os.getenv('OPENAI_API_KEY')

os.environ["OPENAI_API_KEY"] = api_key

if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set. Please set the environment variable before running the script.")

if 'is_authenticated' not in st.session_state:
    st.session_state['is_authenticated'] = False

if 'username' not in st.session_state:
    st.session_state.username = ''

# Splash Screen UI
def display_auth_page():
    st.title('Login to Amanda Chatbot')

    # Directly use the returned value from text_input without setting session_state here
    username = st.text_input('Prolific ID:')

    if st.button('Proceed'):
        if username:  # Check the username directly from the input
            st.session_state.username = username  # Set the session state username
            st.session_state.is_authenticated = True
            st.experimental_rerun()

        else:
            st.warning('Please enter a Prolific ID to proceed.')


    # Display a non-interactive text area with pre-defined text
    text_area_html = """
        <style>
            textarea {
                width: 100%;
                height: 100px;
                background-color: transparent;
                color: black;
                border: 1px solid #ccc;
                padding: 10px;
            }
        </style>
    Please note that in order to interact with Amanda, what you write will be sent to OpenAI in the U.S. and will be stored on their servers for up to 30 days. Thus, please do not provide any identifying information or personal details when interacting with the chatbot.
        </textarea>
        """
    st.markdown(text_area_html, unsafe_allow_html=True)


def display_main_chat():
    ################################### LLM STUFF ####################################

    initial_bot_message = "Hi. I'm Amanda. Can you tell me what has brought you in today?"

    template = """You are a trained psychotherapist called Amanda specialising in working with relationship difficulties. 
    I would like you to respond as a relationship therapist: reflect what the client has said, provide validation and empathy, stay close to what the client says instead of overinterpreting them, and ask follow-up questions designed for you to better understand the situation. 
    Do not provide answers that are too long, only ask one question at a time, and try to maintain a natural conversation like I would have with a therapist.
    The conversations should last between 20-30 minutes and should eventually end up with some relevant suggestions for how to improve the issue but this should only come towards the end of the conversation once the person has had enough time to explore their issue and you have a good understanding of the issue and feel you can offer personalised suggestions for help
        If they do not provide any context, assume you know nothing about their situation and ask them for more information. Avoid asking any information that is identifying (e.g. do not ask addresses, names, or companies where they work).
        Conversation history: {chat_history}
        Current patient utterance: {patient_utterance}
        Only return the helpful response below.
        Helpful response:"""

    prompt = PromptTemplate(
            input_variables=["chat_history", "patient_utterance"], template=template)

    llm = ChatOpenAI(model_name='gpt-4')

    @st.cache_resource
    def get_memory():
        return ConversationBufferMemory(memory_key="chat_history")

    @st.cache_resource
    def get_llm_chain():
        return LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

    bot_icon_seed = 23

    with st.sidebar:
        st.title('<3 Amanda')
        st.markdown('''
        ## About
        ''')
        add_vertical_space(1)
        st.write('Made by Matthew Vowels')

    # Generate empty lists for generated and past.
    # stores AI generated responses
    if 'generated' not in st.session_state:
        st.session_state['generated'] = [initial_bot_message]

    if 'past' not in st.session_state:
        st.session_state['past'] = [' ']

    # Layout of input/response containers
    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()


    if 'input_text' not in st.session_state:
        st.session_state.input_text = ''

    def submit():
        st.session_state.input_text = st.session_state.input_widget
        st.session_state.input_widget = ''
    # User input
    ## Function for taking user provided prompt as input
    def get_text():
        input_text = st.text_input("You: ", "", key="input_widget", on_change=submit)
        return st.session_state.input_text

    ## Applying the user input box
    with input_container:
        user_input = get_text()

    ## Conditional display of AI generated responses as a function of user provided prompts
    with response_container:

        memory = get_memory()
        llm_chain = get_llm_chain()

        if user_input:
            st.session_state.past.append(user_input)
            placeholder = st.empty()

            # Display something in the placeholder
            placeholder.text("Amanda is typing...")

            chat_placeholder = st.empty()
            with chat_placeholder.container():
                a = message(user_input, key='temp', is_user=True, avatar_style='identicon')

                if len(st.session_state['generated']) == 1:
                    print('doing this')
                    message(st.session_state["generated"][0], key='temp_gen', avatar_style='open-peeps', seed=bot_icon_seed)

            response = llm_chain.predict(patient_utterance=user_input).rstrip('\"')
            st.session_state.generated.append(response)
            placeholder.empty()
        if st.session_state['generated']:

            if 'chat_placeholder' in locals():
                chat_placeholder.empty()

            for i in range(len(st.session_state['generated'])-1, -1, -1):  # reverse ordering (old->new  = bottom -> top)
                message(st.session_state["generated"][i], key=str(i), avatar_style='open-peeps', seed=bot_icon_seed)
                if i > 0:
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style='identicon')


    if len(st.session_state.past) < len(st.session_state.generated):
        st.session_state.past.append(None)
    elif len(st.session_state.generated) < len(st.session_state.past):
        st.session_state.generated.append(None)

    # Create a DataFrame with chat history
    df_chat = pd.DataFrame({
        'User Input': st.session_state.past,
        'Model Response': st.session_state.generated
    })

    # Save the DataFrame to a CSV file
    df_chat.to_csv(f'{st.session_state.username}_chat_history.csv', index=False)


# Check for authentication on every run
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False

if st.session_state.is_authenticated:
    display_main_chat()
else:
    display_auth_page()

