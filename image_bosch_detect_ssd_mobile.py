import tensorflow as tf
import numpy as np
import datetime
import time
import os, sys
import cv2
from PIL import Image

import yaml

from glob import glob

try:
    import matplotlib
    matplotlib.use('TkAgg')
finally:
    from matplotlib import pyplot as plt

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util



class TrafficLightClassifier(object):
    def __init__(self, frozen_model_file):
        PATH_TO_MODEL = frozen_model_file
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        return boxes, scores, classes, num


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)



def get_all_labels(input_yaml, riib=False):
    """ Gets all labels within label file

    Note that RGB images are 1280x720 and RIIB images are 1280x736.
    :param input_yaml: Path to yaml file
    :param riib: If True, change path to labeled pictures
    :return: images: Labels for traffic lights
    """
    images = yaml.load(open(input_yaml, 'rb').read())

    for i in range(len(images)):
        images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml), images[i]['path']))
        if riib:
            images[i]['path'] = images[i]['path'].replace('.png', '.pgm')
            images[i]['path'] = images[i]['path'].replace('rgb/train', 'riib/train')
            images[i]['path'] = images[i]['path'].replace('rgb/test', 'riib/test')
            for box in images[i]['boxes']:
                box['y_max'] = box['y_max'] + 8
                box['y_min'] = box['y_min'] + 8
    return images


def detect_label_images(input_yaml, output_folder=None):
    """
    Shows and draws pictures with labeled traffic lights.
    Can save pictures.

    :param input_yaml: Path to yaml file
    :param output_folder: If None, do not save picture. Else enter path to folder
    """

    PATH_TO_LABELS = r'data/bosch_label_map.pbtxt'
    NUM_CLASSES = 14

    frozen_model_file = "./models/bosch_freeze_tf1.3/frozen_inference_graph.pb"
        
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print(category_index)


    # loading models 
    tfc = TrafficLightClassifier(frozen_model_file)

    images = get_all_labels(input_yaml)

    if output_folder is not None:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    for idx, image_dict in enumerate(images[:10]):

        image_path = image_dict['path']
        image_np = cv2.imread( image_path )

        if idx == 0:
            print(image_path)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        boxes, scores, classes, num = tfc.get_classification(image_np)

        vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np, 
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=0.3,
                    line_thickness=8)

        if idx % 10 == 0 and idx > 0:
            print("%d images processed. %s" % ( (idx + 1), image_path   ) )

        image_file = image_path.split("/")[-1]
        cv2.imwrite(  os.path.join( output_folder, image_file )  , image_np )




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(-1)
    label_file = sys.argv[1]
    output_folder = None if len(sys.argv) < 3 else sys.argv[2]
    detect_label_images(label_file, output_folder)