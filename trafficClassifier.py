import tensorflow as tf
import numpy as np
import datetime

import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt 

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


def main():


    PATH_TO_LABELS = r'data/sim_udacity_label_map.pbtxt'
    NUM_CLASSES = 3

    #frozen_model_file = "./models/freeze/frozen_inference_graph.pb"
    frozen_model_file = "./models/sim_freeze_tf1.3/frozen_inference_graph.pb"

    # test_img_dir = "/Users/donchan/Documents/UdaCity/MyProject/bstld/data/train/rgb/train/2015-10-05-11-26-32_bag/jpeg"
    #test_img_dir = "alex-lechner-udacity-traffic-light-dataset/udacity_testarea_rgb"
    test_img_dir = "dataset-sdcnd-capstone/data/sim_training_data/sim_data_capture"
    test_image = "left0546.jpg"
    image_path = os.path.join(test_img_dir,test_image)    
    
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)


    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    

    tfc = TrafficLightClassifier(frozen_model_file)

    boxes, scores, classes, num = tfc.get_classification(image_np)

    print("length of boxes",  len(boxes)) 
    print(scores)
    print(classes)
    print("predicted numbers ", num)
    print("categories", categories)
    print("category index", category_index)


    IMAGE_SIZE = (8,6)
    vis_util.visualize_boxes_and_labels_on_image_array(
                image_np, 
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()


    test_image = "left0030.jpg" #RED
    image_path = os.path.join(test_img_dir,test_image)    
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    boxes, scores, classes, num = tfc.get_classification(image_np)

    print("length of boxes",  np.squeeze(boxes)) 
    print(scores)
    print(classes)
    print("predicted numbers ", num)
    print("categories", categories)
    print("category index", category_index)


    vis_util.visualize_boxes_and_labels_on_image_array(
                image_np, 
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()


    test_image = "left0021.jpg" #YELLOW
    image_path = os.path.join(test_img_dir,test_image)    
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    boxes, scores, classes, num = tfc.get_classification(image_np)

    print("length of boxes",  np.squeeze(boxes)) 
    print(scores)
    print(classes)
    print("predicted numbers ", num)
    print("categories", categories)
    print("category index", category_index)


    vis_util.visualize_boxes_and_labels_on_image_array(
                image_np, 
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()



if __name__ == "__main__":
    main()
