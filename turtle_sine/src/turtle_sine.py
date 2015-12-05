#!/usr/bin/env python

# Import required Python code.
import roslib
#roslib.load_manifest('node_example')
import rospy
import sys
from geometry_msgs.msg import Twist
import math

# Give ourselves the ability to run a dynamic reconfigure server.
#from dynamic_reconfigure.server import Server as DynamicReconfigureServer

# Import custom message data and dynamic reconfigure variables.
#from node_example.msg import node_example_data
#from node_example.cfg import node_example_paramsConfig as ConfigType

# Node example class.
class TurtleSine():
    # Must have __init__(self) function for a class, similar to a C++ class constructor.
    def __init__(self):
        # Get the ~private namespace parameters from command line or launch file.
        #init_message = rospy.get_param('/turtle1/cmd_vel', 'hello')
        pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        rate = rospy.Rate(10)
        # Create a dynamic reconfigure server.
        #self.server = DynamicReconfigureServer(ConfigType, self.reconfigure)
        count = 1

        while not rospy.is_shutdown():
            # Fill in custom message variables with values from dynamic reconfigure server.
            msg = Twist()
            count += 0.01
            msg.linear.x = math.sin(count*10)
            msg.angular.z = 0.2
            # Publish our custom message.
            pub.publish(msg)
            # Sleep for a while before publishing new messages. Division is so rate != period.
            rospy.sleep(0.1)

    # Create a callback function for the dynamic reconfigure server.
    def reconfigure(self, config, level):
        # Fill in local variables with values received from dynamic reconfigure clients (typically the GUI).
        self.message = config["message"]
        self.a = config["a"]
        self.b = config["b"]
        # Return the new variables.
        return config

# Main function.
if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('turtle_sine')
    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        ts = TurtleSine()
    except rospy.ROSInterruptException: pass
