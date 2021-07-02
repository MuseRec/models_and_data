from PIL import Image
from PIL import UnidentifiedImageError
from glob import glob 
import os, json


unidentifed_imgs = []
seen_imgs = 0
for img_file in glob('images/train/train_old/*.jpe'):
    # get the img file name
    file_name = img_file.split(os.path.sep)
    file_name = file_name[-1].split('.jpe')[0]

    try:
        # try to read in the image
        img = Image.open(img_file)

        # save the image as a jpg
        img.save('images/train/train/' + file_name + '.jpg')
        seen_imgs += 1
    except UnidentifiedImageError:
        unidentifed_imgs.append(file_name)

    if seen_imgs % 1000 == 0:
        print(f"Converted {seen_imgs} images")

print(f"{len(unidentifed_imgs)} unidentified images...")

if unidentifed_imgs:
    with open('metadata/iids_unidentified_images.json', 'w') as f_out:
        json.dump(unidentifed_imgs, f_out)
