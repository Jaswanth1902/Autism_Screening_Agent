import kagglehub

# Download latest version
path = kagglehub.dataset_download("fabdelja/asd-screening-data-toddler-child-adoles-adult")

print("Path to dataset files:", path)
