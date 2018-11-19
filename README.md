# Anime-SRGAN

This is a tool for converting low resolution anime to 1080p, with a model architecture largely based from the [SRGAN paper](https://arxiv.org/pdf/1609.04802.pdf).

## Usage

1. Install all python prerequisites (keras, Tensorflow, moviepy, cv2 etc.)

2. Use the create\_high\_res\_video function in the python script (see the example function in anime_converter.py)

3. If you don't have a GPU or would like to run this in the cloud, you can use a Google Colab notebook for video conversion. Notebook Link: https://drive.google.com/file/d/1unT5vtRtJMGEXAhbHbOCQqI387zU7fEF/view?usp=sharing

Note: Conversion speed is roughly 3 frames a second on a GTX 1080 Ti.
