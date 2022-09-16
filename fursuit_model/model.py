import tensorflow as tf
import pathlib
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

MODEL_PATH = os.path.join( pathlib.Path(__file__).parent.resolve(), 'model' ) 

class FursuitModel:
    def __init__(self) -> None:
        self._model = tf.saved_model.load( os.path.join( MODEL_PATH, 'saved_model' ) ) 
        self._category_index = label_map_util.create_category_index_from_labelmap( os.path.join( MODEL_PATH, 'label_map.pbtxt' ) , use_display_name=True )