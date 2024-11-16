from langsmith import Client

# dataset creation (can be done on langsmith web)
dataset_name = "RAG Chatbot Dataset"

client = Client()

dataset = client.create_dataset(dataset_name, description="A small dataset containing QA from RAG chatbot")

# Filter runs to add to the dataset
runs = client.list_runs(
    project_name="fypj-ai-chatbot",
    execution_order=1,
    error=False,
)

for run in runs:
    client.create_example(
        inputs=run.inputs,
        outputs=run.outputs,
        dataset_id=dataset.id,
    )

# running the evaluation on the dataset
