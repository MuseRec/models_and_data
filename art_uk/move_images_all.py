# import tensorflow as tf 
# from tensorflow import keras 

# img_gen = keras.preprocessing.image.ImageDataGenerator(
#     rescale = 1/.255
# )
# generator = img_gen.flow_from_directory(
#     'data/images', class_mode = 'input'
# )


"""
When the compressed image file is unzipped, it is in the incorrect structure to
do any learning with. As it's a large amount of data to move around, we'll do it
via Python.

The structure that we want to end up with is: images/train and images/validation.
We'll rename the folder "ImageExportLukasN" to train and then randomly sample 20%
of the files to move into the 'validation' folder. From there, we can have two
ImageDataGenerator's, one for training and one for validating, that'll feed into
the model.
"""
import os, json, shutil
import random as rand 
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# ensure that it's reproducible 
rand.seed(42)

PERCENTAGE_TO_SAMPLE = 0.20

def move_files(files, substring_to_remove, destination_folder):
    num_of_files = len(files)
    
    missing_files = []
    unidentifed_imgs = []

    count = 0
    for img_file in files:
        # get the path string
        try: 
            path_str = str(img_file.resolve())
        except FileNotFoundError:
            # if we can't find the file, make a note of it and save the information later
            missing_files.append(str(img_file))

            # move to the next image file
            continue

        # split the components by the substring (to remove it)
        split_path = path_str.split(substring_to_remove)

        # join the components together, including removing the leading / (join doesn't work with it)
        new_path = os.path.join('data/images/', destination_folder, split_path[-1][1:])

        # there will almost certainly be a better way to do this, but...
        # open the image in memory
        try: # it may not recognise the image 
            img = Image.open(img_file)
        except (UnidentifiedImageError, PermissionError):
            unidentifed_imgs.append(str(img_file))

            # move to the next image
            continue 

        # make the necessary directory prior to saving the file
        # we set exist_ok = True as multiple images are associated with a single directory
        try: 
            os.makedirs(os.path.dirname(new_path), exist_ok = True)
        except OSError:
            # in some cases, an OS error can be raised due to malformed strings, with a leading :.
            unidentifed_imgs.append(str(img_file))
            continue

        # save the image in its new location
        img.save(new_path)

        # remove the original image
        os.remove(img_file)

        # increment the count for the print statement below
        count += 1

        if count % (num_of_files / 10) == 0:
            print(f"Number of files moved: {count}; Number missing: {len(missing_files)}")
            break

    return missing_files, unidentifed_imgs

def main():
    # get all of the file paths
    image_files = set(Path('data/images/images').rglob('*.jpg'))

    # calculate the number of images that we want to sample
    sample_size = int(PERCENTAGE_TO_SAMPLE * len(image_files))

    # randomly select sample_size number of files from image_files
    random_sample = set(rand.sample(image_files, sample_size))
    assert len(random_sample) == sample_size # sanity check

    # remove the random sample from the image files
    image_files = image_files - random_sample

    # move the validation image set
    print('Moving Validation set...')
    missing_validation, unidentified_imgs_validation = move_files(
        random_sample, 'ImageExportLukasN', 'validation'
    )

    print('Moving Train set...')
    # move the rest of the images to the train directory
    missing_train, unidentified_imgs_train = move_files(
        image_files, 'ImageExportLukasN', 'train'
    )

    if missing_validation:
        print(f"Missing Validation Images: {len(missing_validation)}")
        json.dump(missing_validation, open('data/images/missing_images_validation.json', 'w'))
    
    if unidentified_imgs_validation:
        print(f"Unidentified Validation Images: {len(unidentified_imgs_validation)}")
        json.dump(
            unidentified_imgs_validation, open('data/images/unidentified_imgs_validation.json', 'w')
        )
    
    if missing_train:
        print(f"Missing Train Images: {len(missing_train)}")
        json.dump(missing_train, open('data/images/missing_images_train.json', 'w'))

    if unidentified_imgs_train:
        print(f"Unidentified Train Images: {len(unidentified_imgs_train)}")
        json.dump(unidentified_imgs_train, open('data/images/unidentified_imgs_train.json', 'w'))

    
if __name__ == '__main__':
    main()

