""" 
    Move the image subset from the local folder into the directory within the project
    and create the requisite train/validation splits.
"""
import os
import shutil
import traceback
import json 
import random as rand 

rand.seed(42)

def move_files(source_images, source_folder, destination_folder):
    count = 0
    for img_file in source_images:
        source = source_folder + '/' + img_file
        destination = destination_folder + '/' + img_file
        
        # copy the file to the new location
        try:
            shutil.copy(source, destination)
        except OSError:
            print(traceback.format_exc())
            break

        if count % 1000 == 0:
            print(f"{count} images copied...")

        count += 1

def create_split(split = 0.1, validation_pre_computed = False):
    train_folder = 'data/images/images/train'
    validation_folder = 'data/images/images/validation'
    subset_folder = 'data/images/images/subset'

    # get all of the images
    img_files = set(os.listdir(subset_folder))

    if not validation_pre_computed:
        # calculate the number of images that we want to randomly sample into validation
        sample_size = int(split * len(img_files))

        # randomly select sample_size number of files from image_files
        random_sample = set(rand.sample(img_files, sample_size))
        assert len(random_sample) == sample_size

        # save the names of the files in the validation set (so we can replicate)
        json.dump(
            list(random_sample), 
            open('data/images/images/validation_sample.json', 'w'), 
            ensure_ascii = False, 
            indent = 4
        )
    else:
        random_sample = set(json.load(open('data/images/images/validation_sample.json', 'r')))
    
    # remove the random sample from the image files
    img_files = img_files - random_sample

    # move the random_sample (the validation data) into the validation folder
    print('Moving validation set...')
    move_files(random_sample, subset_folder, validation_folder)

    # move the rest of the images (the training set) into the train folder
    print('Moving training set...')
    move_files(img_files, subset_folder, train_folder)

def main():
    # move_files(
    #     source_images = os.listdir('F:/Dropbox (The University of Manchester)/ArtUK-sample-images'),
    #     source_folder = 'F:/Dropbox (The University of Manchester)/ArtUK-sample-images',
    #     destination_folder = 'data/images/images/subset'
    # )

    create_split()

if __name__ == '__main__':
    main()