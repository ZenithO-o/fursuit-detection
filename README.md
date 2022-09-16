# Zenith's Fursuit Detection Model! :3
## **Intro**
Hey there! This is a project I have been working on a while to detect fursuits in realtime.


### Youtube Video:

*(Will be here eventually)*

### Twitter Post:

*(Ditto)*

### Other links:

*(Ditto pt 2)*

---

## **How to Use**
Two basic scripts are provided, `run_on_images.py` and `run_with_camera.py`. These two files have arguments associated with them.

### `run_on_images.py`
Use this for running the model on individual image files!~

This script will run the fusuit detection model on any image files you give it.
#### Basic usage:

(run `python run_on_images.py --help` to see all usages)

### `run_with_camera.py`
Use this for running on a connected camera!~

This script will continuously run on the camera selected until the script is exited. 
#### Basic usage:

(run `python run_on_camera.py --help` to see all usages.)

---

## **Installation**

oh god, good luck.

I will eventually work on making this process more seamless and accessible for people not comfortable with tensorflow or dealing with custom packages, but for now, here's the best step-by-step instructions I can give:

1. Install all requirements given in `requirements.txt`
2. Follow [this](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) guide for installing the TensorFlow 2 Object Detection API. If you get stuck, there should be plenty of guides (or frustrated Stack Overflow questions) for how to install this requirement
3. ???
4. Question why you did this in the first place.

---

## **Todo**
- Make a clean `requirements.txt` that includes only the bare minimum
- Implement a more accessible method of installing requirements (probably not happening)
- allow for opening of paths containing images in `run_on_images.py`
- Optimize Model usage with parallel image processing using RaggedTensors(?)
- Write the rest of the README