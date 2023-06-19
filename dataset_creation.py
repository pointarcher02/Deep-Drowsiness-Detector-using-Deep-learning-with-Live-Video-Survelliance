import os
import shutil

dataset_path = r'C:\Users\shash\Desktop\Deep Drowsiness Detection Project\mrlEyes_2018_01'

output_folder = r'C:\Users\shash\Desktop\Deep Drowsiness Detection Project\data'
closed_eyes_folder = os.path.join(output_folder, 'Closed_eyes')
open_eyes_folder = os.path.join(output_folder, 'Open_eyes')

os.makedirs(closed_eyes_folder, exist_ok=True)
os.makedirs(open_eyes_folder, exist_ok=True)

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Iterate through the images in the current folder
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        # Check if the image is open or closed
        if filename[18]=='1':
            # Move the open eye image to the Open_eyes folder
            shutil.move(image_path, os.path.join(open_eyes_folder, filename))
        # elif filename[18]=='0':
        #     # Move the closed eye image to the Closed_eyes folder
        #     shutil.move(image_path, os.path.join(closed_eyes_folder, filename))