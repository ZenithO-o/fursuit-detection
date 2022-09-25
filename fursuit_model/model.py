import numpy as np
import os
import pathlib
import six
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

MODEL_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), 'model') 

class FursuitModel:
    def __init__(self) -> None:
        self._model = tf.saved_model.load(os.path.join(MODEL_PATH, 'saved_model')) 
        self._category_index = label_map_util.create_category_index_from_labelmap(os.path.join(MODEL_PATH, 'label_map.pbtxt') , use_display_name=True)

    def run_model(self, img: np.ndarray) -> dict:

        # Convert img np array into usable format for model
        input_tensor = tf.convert_to_tensor(img)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run model
        detections = self._model(input_tensor)

        # Clean up params for detections result
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        return detections
    
    def visualize_detections(self, img: np.ndarray, detections: dict, threshold: float = 0.5, in_place: bool = False) -> np.ndarray|None:
        
        # Make copy for viz utils (viz utils makes in-place changes to np array)
        if in_place:
            img_np_with_detections = img
        else:
            img_np_with_detections = img.copy()
        
        viz_utils.visualize_boxes_and_labels_on_image_array(
            img_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            self._category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=threshold,
            agnostic_mode=False)
        
        if not in_place:
            return img_np_with_detections
    
    def crop_detections(self, img: np.ndarray, detections: dict, threshold: float = 0.5) -> tuple:
        scores = detections['detection_scores']
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']

        indexes = []
        for idx in range(len(scores)):
            if scores[idx] > threshold:
                indexes.append(idx)

        box_final = [boxes[i] for i in indexes]
        class_final = [classes[i] for i in indexes]
        labels = [self.get_class_name(cl) for cl in class_final]
        
        images = []
        for box in box_final:
            xmin = int(box[1] * img.shape[1])
            ymin = int(box[0] * img.shape[0])
            xmax = int(box[3] * img.shape[1])
            ymax = int(box[2] * img.shape[0])
            images.append(img[ymin:ymax , xmin:xmax])
        
        return images, labels
        
        

    def get_class_name(self, class_int):
        if class_int in six.viewkeys(self._category_index):
            return self._category_index[class_int]['name']
        return 'N/A'
        