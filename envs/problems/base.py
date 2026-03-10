import os

from ..utils import download_file_from_google_drive

class BaseProblem:
    DATA_FOLDER = None  # Should be set by subclasses, e.g., Path("data/tsp")
    GOOGLE_SHARED_ID = None  # Should be set by subclasses, e.g., "1bAoMCVDNl_42rdRy1YlwSAvLYeiSdami"
    # Each instance will be a single data sample, e.g., a TSP instance
    def __init__(self, **kwargs):
        pass

    def to(self, device):
        return self

    def evaluate(self, solutions):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def local_search(self, solutions):
        return solutions

    @classmethod
    def get_val_instances(cls, **kwargs) -> dict[str, list]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    @classmethod
    def get_test_instances(cls, **kwargs) -> dict[str, list]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    @classmethod
    def prepare_dataset(cls):
        # Check if data folder exists and has files, if not, download and prepare
        val_folder = cls.DATA_FOLDER / "val"
        test_folder = cls.DATA_FOLDER / "test"
        if not val_folder.exists() or not any(val_folder.iterdir()) or not test_folder.exists() or not any(test_folder.iterdir()):
            print(f"⚠️ Data folder {cls.DATA_FOLDER} is missing or empty. Downloading and preparing datasets...")
            try:
                os.remove(cls.DATA_FOLDER)  # Clear the folder if it exists
            except Exception:
                pass
            root_data_folder = cls.DATA_FOLDER.parent
            os.makedirs(root_data_folder, exist_ok=True)
            zip_path = root_data_folder / "temp_datasets.zip"
            download_file_from_google_drive(cls.GOOGLE_SHARED_ID, str(zip_path))
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(root_data_folder)
            zip_path.unlink()  # Remove the zip file after extraction
            print(f"🎯 Datasets downloaded and prepared at {cls.DATA_FOLDER}.")


