
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

Additionally, to run the model evaluations, these environment variables are also needed.

`LANGCHAIN_API_KEY`
`LANGCHAIN_PROJ_NAME`
`LANGCHAIN_DATASET`


## Steps

Locate the modelWithConvoHist.py file under the modelling folder

Run the file, and use the chatbot as needed.

To run the model evaluations,

1. Ensure that you have created a [LangSmith account](https://www.langchain.com/langsmith)

2. Create a tracing project and change the project name accordingly

3. Fill in the environment variables accordingly, then run the `modelEvalDataset.py` under the `modelEval` folder.

4. Ask the questions and receive an output from model. The input, output, and context will all be stored in the tracing project in LangSmith. This is how a custom dataset can be created (Not the best way, but due to time constraint, is the most efficient way.)

5. Head to LangSmith and locate your project, check on the runs. To create a dataset from those runs: select all runs, add to dataset.

6. Once again, fill in the environment variables accordingly, Then run the `modelEval.py` under the `modelEval` folder. Now you will be able to track how the model has performed on the LangSmith's website.