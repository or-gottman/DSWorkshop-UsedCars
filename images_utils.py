# Data processing
import pandas as pd
import numpy as np



def download_images(imgs_df, dest_dir, limit=-1):
    import os
    import urllib.request
    import pathlib

    download_dir = dest_dir
    pathlib.Path(download_dir).mkdir(parents=True, exist_ok=True)

    successful_downloads, failed_downloads = 0, 0

    for row in imgs_df.itertuples():
        image_url = row.main_picture_url
        car_id = str(row.id)
        img_extension = image_url.split('.')[-1]
        img_save_name = f'{car_id}.{img_extension}'
        if os.path.isfile(download_dir + img_save_name):
            continue
        print('downloading:', img_save_name)
        try:
            urllib.request.urlretrieve(image_url, download_dir + img_save_name)
            successful_downloads += 1
        except Exception as e:
            print("Error on url:", image_url)
            print(e.args)
            failed_downloads += 1

        limit -= 1
        if limit == 0:
            break
    print("Images downloaded:", successful_downloads)
    print("Images failures:", failed_downloads)


def find_dominant_color_v1(pil_img, palette_size=16):
    from PIL import Image

    # Resize image to speed up processing
    img = pil_img.copy()
    img.thumbnail((100, 100))

    # Reduce colors (uses k-means internally)
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=palette_size)

    # Find the color that occurs most often
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    palette_index = color_counts[0][1]
    dominant_color = palette[palette_index*3:palette_index*3+3]

    return tuple(dominant_color)


def find_dominant_color_v2(image):
    from PIL import Image

    # Resizing parameters
    width, height = 150, 150
    image = image.resize((width, height), resample=0)
    # Get colors from image object
    pixels = image.getcolors(width * height)
    # print(list(image.getdata()))
    # print(pixels)
    # Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    # Get the most frequent color
    dominant_color = sorted_pixels[-2][1]
    return tuple(dominant_color)


def convert_rgb_to_names(rgb_tuple):
    from scipy.spatial import KDTree
    from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb

    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return str(names[index])


def remove_background(img_src_path, img_dst_path):

    import cv2
    import numpy as np

    # Read image
    img = cv2.imread(img_src_path)
    hh, ww = img.shape[:2]

    # threshold on white
    # Define lower and upper limits
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)

    # save results
    cv2.imwrite(img_dst_path, result)