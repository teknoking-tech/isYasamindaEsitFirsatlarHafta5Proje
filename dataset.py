import kagglehub

# Download latest version
path = kagglehub.dataset_download("atasaygin/turkey-earthquakes-19152021")

print("Path to dataset files:", path)