import os
from huggingface_hub import HfApi, create_repo

# --- CONFIGURATION ---
REPO_ID = "sattycodes/my-tts-dataset" 
LOCAL_FOLDER = "/home/cloud/StyleTTS2-fine-tuning/Data"
PRIVATE = True
# ---------------------

def upload_large_dataset():
    api = HfApi()
    
    print(f"Creating/Checking Repo: {REPO_ID}...")
    try:
        create_repo(repo_id=REPO_ID, repo_type="dataset", private=PRIVATE, exist_ok=True)
    except Exception as e:
        print(f"Repo check warning: {e}")

    print(f"Starting Robust Upload from {LOCAL_FOLDER}...")
    print("This will handle files in chunks. Please wait...")
    
    try:
        # switched from upload_folder to upload_large_folder
        future = api.upload_large_folder(
            folder_path=LOCAL_FOLDER,
            repo_id=REPO_ID,
            repo_type="dataset",
            ignore_patterns=[".git", ".cache", "__pycache__", "*.zip"],
            num_workers=4  # Adjusts parallel uploads (4 is usually safe)
        )
        
        # In some versions upload_large_folder returns a future, in others it blocks. 
        # Usually it blocks until done.
        
        print("------------------------------------------------")
        print("Success! Dataset and code uploaded.")
        print(f"View here: https://huggingface.co/datasets/{REPO_ID}")
        print("------------------------------------------------")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    upload_large_dataset()

    # hf upload-large-folder sattycodes/my-tts-dataset /home/cloud/StyleTTS2-fine-tuning/Data --repo-type dataset