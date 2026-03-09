
def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download files from Google Drive. Please install it via 'pip install gdown'."
        )
    gdown.download(url, destination, quiet=False)
