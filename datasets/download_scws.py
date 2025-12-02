"""
Script to download Stanford Contextual Word Similarities (SCWS) dataset.
The SCWS dataset contains word pairs with sentence contexts for evaluating 
context-aware word similarity measures.

Reference: Huang et al. (2012) "Improving Word Representations via Global Context 
and Multiple Word Prototypes"
"""

import urllib.request
import os
import zipfile

def download_scws():
    """Download SCWS dataset from alternative sources."""
    
    # Try multiple mirror URLs
    urls = [
        "https://github.com/facebookresearch/SentEval/raw/main/data/downstream/SCWS/SCWS.zip",
        "http://ai.stanford.edu/~ehhuang/SCWS.zip",
        "https://www.cs.toronto.edu/~lczhang/360/data/SCWS.zip"
    ]
    
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(dataset_dir, "SCWS.zip")
    
    for scws_data_url in urls:
        print(f"Trying to download SCWS dataset from {scws_data_url}...")
        print(f"Saving to {zip_path}")
        
        try:
            urllib.request.urlretrieve(scws_data_url, zip_path)
            print("Download complete!")
            
            # Extract the zip file
            print("Extracting SCWS.zip...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                print(f"Extraction complete! Files extracted to {dataset_dir}")
                
                # Clean up zip file
                os.remove(zip_path)
                print("Cleaned up zip file.")
                
                # Check what was extracted
                scws_dir = os.path.join(dataset_dir, "SCWS")
                if os.path.exists(scws_dir):
                    print(f"\nExtracted files in SCWS/:")
                    for file in os.listdir(scws_dir):
                        print(f"  - {file}")
                
                return True
                
            except Exception as e:
                print(f"Error extracting SCWS dataset: {e}")
                return False
                
        except Exception as e:
            print(f"Failed: {e}")
            if scws_data_url == urls[-1]:
                print("\nAll download sources failed.")
                print("Alternative: You can manually download from:")
                print("  http://ai.stanford.edu/~ehhuang/SCWS.zip")
                print(f"  and extract to {dataset_dir}")
                return False
            else:
                print("Trying next source...\n")
                continue
    
    return False

if __name__ == "__main__":
    download_scws()

