import requests
import zipfile
import io
import os
import numpy as np
from tqdm import tqdm

def download_file(url, local_filename):
    """
    Downloads a large file from a URL in chunks and saves it locally
    with a progress bar.
    """
    print(f"Downloading {url} to {local_filename}...")
    try:
        # Start the request with streaming
        r = requests.get(url, stream=True)
        r.raise_for_status() # Check for download errors
        
        # Get total file size from headers
        total_size = int(r.headers.get('content-length', 0))
        
        # Open the local file in 'write-binary' mode
        with open(local_filename, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=local_filename
        ) as pbar:
            # Write the file in 8KB chunks
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
                
        print("Download complete.")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        # Clean up the partial file if download failed
        if os.path.exists(local_filename):
            os.remove(local_filename)
        return False

def load_vectors_from_local_zip(zip_path, max_words=None):
    """
    Loads fastText word vectors from a local .zip file.
    
    Args:
        zip_path (str): The file path to the local .zip file.
        max_words (int, optional): Load only the top N words. Defaults to None (all words).
    
    Returns:
        dict: A dictionary mapping words to their numpy vector.
    """
    print(f"Loading vectors from {zip_path}...")
    try:
        # 1. Open the local zip file
        with zipfile.ZipFile(zip_path) as z:
            # 2. Get the name of the .vec file inside the zip
            vec_filename = z.namelist()[0]
            print(f"Found vector file in zip: {vec_filename}")
            
            # 3. Open the .vec file within the zip
            with z.open(vec_filename) as f:
                # Wrap in TextIOWrapper to read as text
                fin = io.TextIOWrapper(f, encoding='utf-8', newline='\n', errors='ignore')
                
                # Read the first line (n, d)
                n, d = map(int, fin.readline().split())
                if max_words:
                    n = min(n, max_words)
                
                data = {}
                
                # Use tqdm for a loading progress bar
                for i, line in enumerate(tqdm(fin, total=n, desc="Loading vectors")):
                    if i >= n:
                        break
                        
                    tokens = line.rstrip().split(' ')
                    word = tokens[0]
                    data[word] = np.asarray(tokens[1:], dtype='float32')
                    
                return data

    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file. It may be corrupted.")
        return None
    except FileNotFoundError:
        print(f"Error: File not found at {zip_path}")
        return None

# --- Example Usage ---

# The URL of the file you want to download
url_to_download = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"

# Get the filename from the URL (e.g., "wiki-news-300d-1M.vec.zip")
local_filename = url_to_download.split('/')[-1]

# --- Step 1: Download the file (if it doesn't already exist) ---
if not os.path.exists(local_filename):
    print(f"{local_filename} not found.")
    # Call the download function
    download_file(url_to_download, local_filename)
else:
    print(f"Found {local_filename} locally. Skipping download.")

# --- Step 2: Load the vectors from the local file ---
# if os.path.exists(local_filename):
#     # Load only the top 50,000 words as a quick example
#     word_vectors = load_vectors_from_local_zip(local_filename, max_words=50000)
    
#     if word_vectors:
#         print(f"\nSuccessfully loaded {len(word_vectors)} word vectors.")
#         # Check a vector
#         if 'king' in word_vectors:
#             print("Vector for 'king' (first 5 dims):", word_vectors['king'][:5])
# else:
#     print("File download failed. Cannot load vectors.")