import os
os.environ["KAGGLE_CONFIG_DIR"] = "/Users/junsu/Desktop/ktb/개인 프로젝트(2주)/.kaggle"

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Orange Diseases 데이터셋 다운로드
dataset = "kritikseth/fruit-and-vegetable-image-recognition"
download_path = "fruit-and-vegetable-image"

print(f"Downloading {dataset} dataset...")
api.dataset_download_files(dataset, path=download_path, unzip=True)

print("Download completed!")