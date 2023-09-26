# TherapyApp
Simple streamlit therapy app with langchain.

```amanda_chatbot.py``` runs a local model
```gpt_chatbot.py``` runs using the openAI API (and requires an API key)

To run ```amanda_chatbot.py```, create a models directory and download a .bin file from e.g. https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main

Run one of the scripts with e.g.:

```bash
streamlit run gpt_chatbot.py```

