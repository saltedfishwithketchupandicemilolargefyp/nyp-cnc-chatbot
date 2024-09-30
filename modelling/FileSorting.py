import os
import shutil

# Define source and target directories
source_dir = 'CNC chatbot'  # Source folder where your files are located
target_dir = 'data'         # Target folder to store the separated files

# Create subdirectories in the target 'data' folder for each file type
file_types = ['pdf_files', 'docx_files', 'xlsx_files', 'pptx_files']

for file_type in file_types:
    os.makedirs(os.path.join(target_dir, file_type), exist_ok=True)

# Loop through the source directory and subdirectories
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # Get full file path
        file_path = os.path.join(root, file)
        
        # Separate files by extension and copy to the appropriate folder in the target directory
        if file.endswith('.pdf'):
            shutil.copy(file_path, os.path.join(target_dir, 'pdf_files', file))
        elif file.endswith('.docx'):
            shutil.copy(file_path, os.path.join(target_dir, 'docx_files', file))
        elif file.endswith('.xlsx'):
            shutil.copy(file_path, os.path.join(target_dir, 'xlsx_files', file))
        elif file.endswith('.pptx'):
            shutil.copy(file_path, os.path.join(target_dir, 'pptx_files', file))

print("Files have been copied to the respective folders.")
