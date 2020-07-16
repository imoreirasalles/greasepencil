import greasepencil

# create list of files to be processed
extensions = (".jpg", ".jpeg")
image_files = [
    image_file
    for image_file in os.listdir(f"{source}")
    if str(image_file).endswith(tuple(extensions))
]

# call function
for i in image_files:
    try:
        greasepencil(i)
    except Exception as e:
        print(str(e))