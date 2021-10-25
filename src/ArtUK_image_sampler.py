import pandas as pd
import os
import glob
import json
import shutil
from pathlib import Path
import PIL
from pandas.errors import ParserError
import numpy as np
from PIL import Image, UnidentifiedImageError

# Written by Mama.

#Load the sampled csv file into a df
def get_data():
    image_ids = pd.read_csv("../data/Art-UK/ArtUK_main_sample.csv", usecols=['Filename'])
    print(len(image_ids))
    print(image_ids.head(5))
    return image_ids

#Create a list with all the image IDs
def get_image_ids():
    image_list = []

    for f in glob.glob('/Users/user/Desktop/ImageExportLukasN/**/*.jpg', recursive = True):
        # f_name = f.split('/')[-1] < if we wanted to split the paths and append
        image_list.append(f)

    print(len(image_list))
    print(image_list[0:10])
    return image_list

#match the image IDs to the IDs stored in the csv (these are the IDs we have sampled beforehand)
def ID_match(image_ids, image_list):
    metadata_filenames = set(image_ids['Filename'].values.tolist())
    
    print(len(metadata_filenames))
    return metadata_filenames

# all necessary steps to save the images we matched into a new directory
def copy_files(image_list, metadata_filenames):
    num_of_files = len(image_list)
    
    #create directories to save missing or unidentified images
    missing_files = []
    unidentified_imgs = []

    count = 0
    for img_file in image_list:
        # test that path exists (i.e. file exists)
        # if not, save as a missing file
        if not os.path.isfile(img_file):
            missing_files.append(img_file)
            continue
        # try:
        #     # path_str = str(img_file.resolve())
        
        # except FileNotFoundError:

        #     missing_files.append(str(img_file))

        #     continue

        img_file_id = img_file.split('/')[-1]

        print(img_file_id)
        
        # get if img_file is in the match_ids
        if img_file_id in metadata_filenames:
            # create new path to save image
            new_path = '/Users/user/Desktop/ArtUK-sample-images/' + img_file_id

            try: 
                img = Image.open(img_file)
            except (UnidentifiedImageError, PermissionError):
                # if there's error (can't read or permission) then make a record of the image
                unidentified_imgs.append(img_file_id)
                continue # skip over this image
            
            # save the image in the new location
            img.save(new_path)

        

    return missing_files, unidentified_imgs
            
    

def main():
    image_IDs = get_data()

    image_list = get_image_ids()

    metadata_filenames = ID_match(image_IDs, image_list)

    missing_files, unidentified_imgs = copy_files(image_list, metadata_filenames)

    # saving the missing and unidentified images.
    if missing_files:
        print(f"Missing files detected: {len(missing_files)}")
        json.dump(missing_files, open('../data/Art-UK/missing_files.json', 'w'))

    if unidentified_imgs:
        print(f"Unidentified images detected: {len(unidentified_imgs)}")
        json.dump(unidentified_imgs, open('../data/Art-UK/unidentified_imgs.json', 'w'))




if __name__ == '__main__':
    main()