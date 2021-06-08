# poc_ml_compression

> Proof of concept to see if a ML model can be used to compress a video file.

The ML model file itself would then be used as the video file which would
require its own codec to encode (train the ML model) and decode (extract frames
using only frame number).

The results of the experiment can be shown below:

## Input

<img src="training/frame0.png">

## Output

<img src="frames/output-0.png">

The system works by training the model to predict an entire video frame given
only a frame number (0 through FRAME_WIDTH * FRAME_HEIGHT).

> This idea has merit but model fitting and prediction performance is very slow.
