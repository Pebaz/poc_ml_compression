from keras.models import load_model
from PIL import Image

IMAGE_WIDTH = 1856
IMAGE_HEIGHT = 1068

model = load_model('model')

print('here')

for frame in range(10):
    print('OUTPUTTING: Frame ->', frame)
    # Plus 1 here to avoid 0
    raw_pixel_data = model.predict((frame + 1,))[0]
    image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT))
    output_pixels = [
        (
            (pixel >> 16) & 0xFF,
            (pixel >> 8) & 0xFF,
            pixel & 0xFF
        )
        for pixel in map(lambda rgb: int(rgb * 0xFFFFFF), raw_pixel_data)
    ]

    image.putdata(output_pixels)

    image.save(f'frames/output-{frame}.png')

