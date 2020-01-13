from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import os

class TLClassifier(object):
    def __init__(self):
        print(os.getcwd())
        model = 'frozen_models/simulator/frozen_inference_graph.pb'
        self.detection_graph = self.load_graph(model)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        
    def load_graph(self, graph_file):
        """ Loads frozen inference graph, the pretrained model """
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    
    def filter_boxes(self, min_score, boxes, scores, classes):
        """ Return boxes with a confidence >= `min_score` """
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs,  ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes




    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #Traffic light color prediction
        traffic_light_state = TrafficLight.UNKNOWN
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims( np.asarray(image_rgb, dtype=np.uint8), 0)

        with tf.Session(graph = self.detection_graph) as sess:
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                 feed_dict={self.image_tensor: image_np})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)
            confidence_cutoff = 0.5
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
        if len(scores) <= 0:
            traffic_light_class_id = 4
            print("traffic_light_state = %s", traffic_light_state)
            return traffic_light_state

        # traffic light detected, return light state classification
        traffic_light_class_id = int(classes[np.argmax(scores)])
        
        if traffic_light_class_id == 1:
            print("Traffic Light GREEN")
            traffic_light_state = TrafficLight.GREEN
        elif traffic_light_class_id == 2:
            print("Traffic Light RED")
            traffic_light_state = TrafficLight.RED
        elif traffic_light_class_id == 3:
            print("Traffic Light YELLOW")
            traffic_light_state = TrafficLight.YELLOW

        return traffic_light_state
