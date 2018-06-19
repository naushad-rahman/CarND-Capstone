from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import rospy

class TLClassifier(object):
    def __init__(self):
        self.detection_graph = tf.Graph()
        graph_path = './graph/frozen_inference_graph.pb'
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    # expect rgb.
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # Convert image format.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                green_score = 0 # id=1
                red_score = 0 # id=2
                yellow_score = 0 # id=3
                detected = False

                # FIXME: Imporve logic.
                for cl, score in zip(classes, scores):
                    if score > 0.5:
                        detected = True
                        if cl == 1:
                            green_score = green_score + score
                        if cl == 2:
                            red_score = red_score + score
                        if cl == 3:
                            yellow_score = yellow_score + score

                if detected:
                    sums = np.array([green_score, red_score, yellow_score])
                    idx = np.argsort(sums)[-1]
                    rospy.loginfo('detected light idx: {}'.format(idx))

                    if idx == 0:
                        return TrafficLight.GREEN
                    if idx == 1:
                        return TrafficLight.RED
                    if idx == 2:
                        return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN
