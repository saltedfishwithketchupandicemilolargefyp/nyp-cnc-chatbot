
# FYPJ CNC Chatbot

The aim is to build an AI Chatbot that is context specific to the NYP Data Classification Systems. 
To facilitate and answer potential queries staff have about different CNC related questions, instead of having to search and refer to the documentation files to find the answers. 



## Installation

1. Git Clone the repository (Branch: **main**)

2. Install [Python Version 3.10.2](https://www.python.org/downloads/release/python-3102/)

      a. Open the installer and ensure to check the box (add to PATH) on the bottom of the installer screen.

      b.	Once done, open the cloned git hub repository.

      c.	Ctrl + Shift + P and search 'Python: Select Interpreter'
  
      d.	Find and select Python 3.10.2 


3. Create a virtual environment

   Open a terminal (powershell)
   
```bash
  python -m venv .venv
```

4. Activate the venv

   To activate the venv, create a new terminal, a command prompt terminal (NOT powershell)
      
```bash
  cd .venv\Scripts
  activate
  cd ..
  cd ..
```

5. Install requirements

```bash
  pip3 install -r requirements.txt
```


## Running the Model Interface (Frontend & Backend)

Run app.py and streamlit_app.py

```bash

python app.py
streamlit run streamlit_app.py

```


## Running the Model in Terminal 
Navigate to modelling folder > modelWithConvoHist.py
Right Click code area and Run Python > Run Python File in Terminal


    
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`OPENAI_API_KEY`

To run the model evaluation, you will additionally need this environment variables

`LANGCHAIN_API_KEY`

