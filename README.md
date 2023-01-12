# Zenith's Fursuit Detection Model! :3
## **Introduction**
Hey there! This is a project I have been working on a while to detect fursuits in realtime.


### Youtube Video:

*(Will be here eventually)*

### Twitter Post:

*(Ditto)*

### Other links:

## **How to Use**
Two basic scripts are provided, `run_on_images.py` and `run_with_camera.py`. These two files have arguments associated with them.

### `run_on_images.py`
Use this for running the model on individual image files!~

This script will run the fursuit detection model on any image files you give it.
#### Basic usage:

(run `python run_on_images.py --help` to see all usages)

### `run_with_camera.py`
Use this for running on a connected camera!~

This script will continuously run on the camera selected until the script is exited. 
#### Basic usage:

(run `python run_on_camera.py --help` to see all usages.)


### Other Info
`fursuit_model/model.py` contains the `FursuitModel` class used to run the model. It should be fairly straight-forward to use, type hinted and all. 

I would make it into a package, but me lazy :)



## **Installation**

oh god, good luck. This project requires ~2.2GB of space. Requires Python >3.10 (probably can use older but idk)

I will eventually work on making this process more seamless and accessible for people not comfortable with tensorflow or dealing with custom packages, but for now, this is the best I can offer... (I will gladly accept any help from others making this more accessible for people to use)

Here's the best step-by-step instructions I can give:

1. Install all requirements given in `requirements.txt`
2. Follow [this link for a guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) on installing the TensorFlow 2 Object Detection API. If you get stuck, there should be plenty of guides (or frustrated Stack Overflow questions) for how to install this requirement.
3. ???
4. If you run into some dependency errors, try running `pip install --no-deps -r requirements.txt` after finishing the object detection installation guide.
5. Question why you did this in the first place.

AFAIK, there are some [issues](https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal) with installing the `object_detection` dependencies (yay version conflicts and backwards compatibility), so if this process is not working, I am sure there are modern guides to installing the object detection library for tensorflow.


## **Todo (in order of importance)**
- Docstrings for `fursuit_model/model.py`
- Convert `fursuit_model/model.py` into a package for easier usage
- Allow for opening of paths containing images in `run_on_images.py`
- Implement a more accessible method of installing requirements (probably Dockerize-ing it, but I have no idea how to use Docker)
- Optimize Model usage by allowing parallel image processing using RaggedTensors(?)
- Implement a tflite version of the model that can be used with more hardware. (This is has been frustrating every time I've tried it, so I don't care to do it atm)