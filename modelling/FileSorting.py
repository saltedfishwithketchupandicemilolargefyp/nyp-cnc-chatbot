# import required libraries for file operations
import os 
import shutil

# define the main directories
source_dir = 'CNC chatbot'  # where the original mixed files are stored
target_dir = 'data'         # where we'll organize files by type

# list of folders we need to create for different file types
# each file type will have its own dedicated folder
file_types = ['pdf_files', 'docx_files', 'xlsx_files', 'pptx_files']

# create subdirectories for each file type
# exist_ok=True prevents errors if directories already exist
for file_type in file_types:
    os.makedirs(os.path.join(target_dir, file_type), exist_ok=True)

# walk through all directories and subdirectories in the source folder
# os.walk returns root (current dir), dirs (subdirs), and files in current dir
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # construct the full path for each file
        file_path = os.path.join(root, file)
        
        # check file extension and copy to appropriate target folder
        if file.endswith('.pdf'):
            shutil.copy(file_path, os.path.join(target_dir, 'pdf_files', file))
        elif file.endswith('.docx'):
            shutil.copy(file_path, os.path.join(target_dir, 'docx_files', file))
        elif file.endswith('.xlsx'):
            shutil.copy(file_path, os.path.join(target_dir, 'xlsx_files', file))
        elif file.endswith('.pptx'):
            shutil.copy(file_path, os.path.join(target_dir, 'pptx_files', file))

print("Files have been copied to the respective folders.")