"""
Loading(Fanout) -> Fitting(Queue)

Loading:
    Load from file
        Convert to layer
    Load from cache

Fitting:
    Fit to model


Outputting(Fanout)

Outputting:
    Save to frames
"""

import os
import pickle
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from keras.layers import *
from keras.models import *
from PIL import Image
from para import (
    Query,
    PoisonPill,
    QueueFinished,
    ExceptionRaised,
    QueueGetItemTimeout
)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
QUEUE_TIMEOUT = 10
IMAGE_WIDTH = 1856
IMAGE_HEIGHT = 1068
NUM_FITS_PER_FRAME = 10


def task_loading(sequence, filename, fitting_queue):
    def load_frame_to_layer(filename):
        image = Image.open(filename)
        layer = []
        for x in range(image.width):
            for y in range(image.height):
                r, g, b = image.getpixel((x, y))
                rgb = (r << 16) | (g << 8) | b
                layer.append(float(rgb) / 0xFFFFFF)
        return layer

    cached_frame = Path('training-cache') / Path(filename).name
    if cached_frame.exists():
        print('CACHED', cached_frame.name)
        layer = pickle.load(cached_frame.open('rb'))
    else:
        print('Not in cache', cached_frame.name)
        layer = load_frame_to_layer(filename)
        pickle.dump(layer, cached_frame.open('wb'))

    fitting_queue.put((sequence, layer))


def task_fitting(me):
    print('FITTING')
    try:
        # Load the model if possible
        cached_model = Path('model')

        # Create the model
        if not [*cached_model.iterdir()]:
            print('FITTING: Model was not cached, creating...')
            inputs = Input(shape=(1,))
            model_shape = IMAGE_WIDTH * IMAGE_HEIGHT
            # outputs = Dense(model_shape, activation='sigmoid')(inputs)
            # outputs = Dense(model_shape, activation='softmax')(inputs)

            hidden_layer1 = Dense(units=20, activation='relu')(inputs)
            hidden_layer2 = Dense(units=20, activation='relu')(hidden_layer1)
            outputs = Dense(model_shape, activation='relu')(hidden_layer2)
            model = Model(inputs=inputs, outputs=outputs)

            model.compile(loss='msle', optimizer='adam')
            # model.compile(loss='mse', optimizer='adam')
            # model.compile(loss='cosine_similarity', optimizer='adam')
            model.save('model')

        # Have to load the model
        else:
            print('FITTING: Model was cached')
            model = load_model('model')

        while True:
            try:
                item = me.queue.get(timeout=QUEUE_TIMEOUT)
                if isinstance(item, PoisonPill):
                    if isinstance(item, QueueFinished):
                        print('FITTING: Queue finished')
                        break
                    elif isinstance(item, ExceptionRaised):
                        raise item.exception

                sequence, layer = item
                print('FITTING:', 'Frame ->', sequence)

                # model.fit((sequence,), (layer,))
                model.fit(
                    [sequence] * NUM_FITS_PER_FRAME,
                    [layer] * NUM_FITS_PER_FRAME
                )

            except QueueGetItemTimeout:
                print('FITTING: Queue read timed out')
                break

        print('FITTING: Saving model')
        model.save('model')
    except Exception as e:
        print(f'FITTING: Something went wrong: {e}')


fitting = Query(target=task_fitting, daemon=True)
fitting.start()

threads = []
with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
    frames = [*Path('training').iterdir()]
    frames.sort(key=lambda x: int(Path(x).stem.replace('frame', '')))

    for i, frame in enumerate(frames[:10]):
        # Plus 1 here to avoid 0
        threads.append(pool.submit(task_loading, i + 1, frame, fitting.queue))

for thread in as_completed(threads):
    thread.result()

fitting.queue.put(QueueFinished())
fitting.join()


# import moviepy.video.io.ImageSequenceClip

# images = [str(p) for p in Path('training').iterdir()]
# clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=4)
# clip.write_videofile('my_video.mp4')


# model.fit()


