import os

from PIL import Image


def concatenate_images(list_images: list, output_file: str):
    
    images = []
    for filepath in list_images:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)
        images.append(Image.open(filepath))

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_img = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset,0))
        x_offset += img.size[0]
    
    new_img.save(output_file)

    