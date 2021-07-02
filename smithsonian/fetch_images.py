import json 
import requests
import mimetypes
import time
from glob import glob

def read_metadata():
    data = []
    for json_file in glob('metadata/*.json'):
        # read in the newline separated json file
        with open(json_file, 'r', encoding = 'UTF-8') as f_in:
            for line in f_in:
                data.append(json.loads(line))

    return data

def _check_all_ids_are_unique(data):
    assert len(data) == len(set([val['id'] for val in data]))

def _check_all_media_has_only_one_image(data):
    media_lengths = set([
        val['content']['descriptiveNonRepeating']['online_media']['mediaCount']
        for val in data
    ])
    assert len(media_lengths) == 1
    assert int(tuple(media_lengths)[0]) == 1

def download_images(data):
    not_okay_status_codes = []
    for idx, meta in enumerate(data):
        # if the index is a multiple of 1000, then sleep for 30 seconds
        # (the numbers are arbitary, just don't want to hit rate limits)
        if (idx % 1000) == 0:
            print(f"{idx} - Sleeping...")
            time.sleep(30)

        # get the id of the item
        iid = meta['id']

        # get the url (assumption that there's only one, see function above)
        url = meta['content']['descriptiveNonRepeating']['online_media']['media'][0]['content']

        # get the content on the other side of the url
        response = requests.get(url)

        if response.status_code == 200:
            # figure out the extension for the image
            extension = mimetypes.guess_extension(
                response.headers['content-type'])
            
            # file name
            file_name = iid + extension

            # save the image
            with open('images/' + file_name, 'wb') as img_out:
                img_out.write(response.content)
        else:
            not_okay_status_codes.append(iid)
    
    return not_okay_status_codes

if __name__ == '__main__':
    # read all of the metadata
    data = read_metadata()

    # sanity check that all ids are unique
    _check_all_ids_are_unique(data)

    # sanity check: do all entries online have one media source?
    _check_all_media_has_only_one_image(data)

    # fetch the images
    missing = download_images(data)

    print(f"Number of missing images: {len(missing)}")
    
    # write those missing to file
    with open('metadata/iids_missing_images.json', 'w') as m_out:
        json.dump(missing, m_out)

