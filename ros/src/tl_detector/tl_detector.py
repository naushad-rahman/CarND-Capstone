#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from styx_msgs.msg import TrafficLightArray, TrafficLight, GlobalTrafficLightMessage
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

from geometry_msgs.msg import PoseStamped, Pose
import cv2
import yaml
import math
import time
import numpy as np


STATE_COUNT_THRESHOLD = 1

# gamma correction function used to reduce high sun exposure 
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table) 


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints_2D = None
        self.waypoint_TREE = None
        self.sim_testing = False 
        self.gamma_correction = False
        self.count = 0
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', GlobalTrafficLightMessage, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier =TLClassifier(0.01, 0.5, False)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
       

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        
        if not self.waypoints_2D:
            self.waypoints_2D = [[waypoint.pose.pose.position.x,waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints ]
            self.waypoint_TREE = KDTree(self.waypoints_2D)

    def traffic_cb(self, msg):
        self.lights = msg.lights


    def _pass_threshold(self):
        if self.state == TrafficLight.YELLOW:
            return self.state_count >= STATE_COUNT_THRESHOLD - 1
        return self.state_count >= STATE_COUNT_THRESHOLD


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
       
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        # if self.state != state:
        #     self.state_count = 0
        #     self.state = state
        # elif self.state_count >= STATE_COUNT_THRESHOLD:
        #     self.last_state = self.state
        #     light_wp = light_wp if state == TrafficLight.RED else -1
        #     self.last_wp = light_wp
        #     self.upcoming_red_light_pub.publish(Int32(light_wp))
        # else:
        #     self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        # self.state_count += 1
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self._pass_threshold():
            self.last_wp = light_wp
            msg = self._prepare_result_msg(state, light_wp)
            rospy.loginfo(msg)
            self.upcoming_red_light_pub.publish(msg)
        else:
            msg = self._prepare_result_msg(self.state, self.last_wp)
            rospy.loginfo(msg)
            self.upcoming_red_light_pub.publish(msg)
            self.state_count += 1

    def _prepare_result_msg(self, tl_state, tl_stop_waypoint):
        tl_result = GlobalTrafficLightMessage()
        tl_result.state = tl_state
        tl_result.waypoint = tl_stop_waypoint

        return tl_result






    def get_closest_waypoint(self, x,y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_TREE.query([x,y],1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if self.light_classifier is None: # Stop if classifer is not yet ready
            return TrafficLight.RED
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        processed_img = cv_image[0:600, 0:800] # was [20:400, 0:800]

        #TODO use light location to zoom in on traffic light in image
        #Prepare image for classification
        if self.sim_testing: #we cut 50 pixels left and right of the image and the bottom 100 pixels
            width, height, _ = cv_image.shape
            x_start = int(width * 0.10)
            x_end = int(width * 0.90)
            y_start = 0
            y_end = int(height * 0.85)
            processed_img = cv_image[y_start:y_end, x_start:x_end]
        else:   # real-case testing. Reduce image size to avoid light reflections on hood.
            processed_img = cv_image[0:600, 0:800] # was [20:400, 0:800]           

        #Convert image to RGB format
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        #Get classification

        #initialize light_state to unknown by default
        light_state = TrafficLight.UNKNOWN
        light_state_via_msg = None

        #get the ground truth traffic light states through the traffic light messages
        # for tl in self.lights:
        #     rospy.loginfo(light)
        #     dist = math.sqrt((tl.pose.pose.position.x - light.position.x)**2 + (tl.pose.pose.position.y - light.position.y)**2)
        #     if (dist < 50): #means we found the light close to the stop line
        #         light_state_via_msg = tl.state
        #         break #no need to parse other lights once light was found

        #detect traffic light position (box) in image
        #convert image to np array
        img_full_np = self.light_classifier.load_image_into_numpy_array(processed_img)
        
        #apply gamma correction to site testing if parameter set in launch file.
        if (self.gamma_correction == True):
            img_full_np = adjust_gamma(img_full_np, 0.4)
        
        # if simulator, we apply detection and classification separately
        
        unknown = False

        if self.sim_testing:
            # find traffic light in image.
            b = self.light_classifier.get_localization(img_full_np)
            print(b)
            # If there is no detection or low-confidence detection
            if np.array_equal(b, np.zeros(4)):
               print ('unknown')
               unknown = True
            else:    #we can use the classifier to classify the state of the traffic light
               img_np = cv2.resize(processed_img[b[0]:b[2], b[1]:b[3]], (32, 32))
               self.light_classifier.get_classification(img_np)
               light_state = self.light_classifier.signal_status
        else:
            print("Get in Localization-Classification")
            b, conf, cls_idx = self.light_classifier.get_localization_classification(img_full_np, visual=False)
            print("Get out of Localization-Classification")
            if np.array_equal(b, np.zeros(4)):
                print ('unknown')
                unknown = True
            else:
                #light_state = cls_idx
                if cls_idx == 1.0:
                    print('Green', b)
                    light_state = TrafficLight.GREEN
                elif cls_idx == 2.0:
                    print('Red', b)
                    light_state = TrafficLight.RED
                elif cls_idx == 3.0:
                    print('Yellow', b)
                    light_state = TrafficLight.YELLOW
                elif cls_idx == 4.0:
                    print('Unknown', b)
                    light_state = TrafficLight.UNKNOWN
                else:
                    print('Really Unknown! Didn\'t process image well', b)
                    light_state = TrafficLight.UNKNOWN
                    
        #check prediction against ground truth
        if self.sim_testing:
            rospy.loginfo("Upcoming light %s, True state: %s", light_state, light_state_via_msg)
            #compare detected state against ground truth for (simulator only)
            if not unknown:
                self.count = self.count + 1
                filename = "sim_image_" + str(self.count)
                if (light_state == light_state_via_msg):
                   self.tp_classification = self.tp_classification + 1
                   #filename = filename + "_good_" + str(light_state) + ".jpg"
                else:
                   filename = filename + "_bad_" + str(light_state) + ".jpg"
                    #cv2.imwrite(filename, cv_image)
                self.total_classification = self.total_classification + 1
                accuracy = (self.tp_classification / self.total_classification) * 100
                if self.count % 20 == 0:
                    rospy.loginfo("Classification accuracy: %s", accuracy)
        else: #site testing
            self.count = self.count +1
            #filename = "site_image_" + str(self.count) + str(light_state) + ".jpg"
            #cv2.imwrite(filename, cv_image)
            
        return light_state

    def get_light_state2(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # if(not self.has_image):
        #     self.prev_light_loc = None
        #     return False

        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
       
        if (self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i,light in enumerate(self.lights):
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0],line[1])
                d = temp_wp_idx - car_wp_idx
                if d >=0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx ,state
        
        return -1,TrafficLight.UNKNOWN

        
if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
