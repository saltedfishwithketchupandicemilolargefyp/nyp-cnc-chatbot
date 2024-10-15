
# Project Title

A brief description of what this project does and who it's for


## Installation

1. Git Clone the repository (Branch: **yq_model**)

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



    
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`OPENAI_API_KEY`


## Steps

Locate the modelWithConvoHist.py file under the modelling folder

Run the file, and use the chatbot as needed.
