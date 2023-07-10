import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def img_callback(img_msg):
    bridge = CvBridge()
    conver_img = bridge.imgmsg_to_cv2(img_msg, "mono8")
    cv2.imwrite("img/test.jpg", conver_img)
    rospy.signal_shutdown("Message received")

rospy.init_node('img_saver', anonymous=False)
# yamlPath = rospy.get_param("~config_path", "/home/plus/Work/plvins_ws/src/PL-VINS/config/feature_tracker/mtuav_config.yaml")
# with open(yamlPath,'rb') as f:
#     params = yaml.load(f, Loader=yaml.FullLoader)
#     point_params = params["point_feature_cfg"]
#     camera_params = params["camera_cfg"]

# my_point_extract_model = MyPointExtractModel(point_params)  # 利用参数文件建立自定义点特征模型
# my_point_match_model = MyPointMatchModel(point_params)

# CameraIntrinsicParam = PinholeCamera(
#     fx = 461.6, fy = 460.3, cx = 363.0, cy = 248.1, 
#     k1 = -2.917e-01, k2 = 8.228e-02, p1 = 5.333e-05, p2 = -1.578e-04
#     )  

image_topic = "/mtuav/stereo_down0/image_raw"

sub_img = rospy.Subscriber(image_topic, Image, img_callback, 
                            queue_size=1) 
rospy.spin()