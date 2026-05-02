import numpy as np
import time
import cv2
import threading
import sys
from scipy.spatial.transform import Rotation as R
import math

from exercises.ex3_motion_planner import MotionPlanner3D as MP


class _RaceMotionPlanner3D(MP):
    def plot(self, obs, path_waypoints, trajectory_setpoints):
        # Disable plotting in the live controller.
        return


# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate


class MyAssignment:
    def __init__(self):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.latest_feed = None
        self.latest_feed_lock = threading.Lock()
        self.img_width = 300 #the img is a 300x300 square
        self.cam_to_body = np.array([[0,0,1], #rotation matrix from camera origin orientation to body origin orientation
                                    [-1,0,0],
                                    [0,-1,0]])
        # self.latest_obstacles_corners = None
        self.triangulation_list = [] #4 corners saved every 10 cm #NB : maybe later I don't need a whole list but just the last 2 values ?
        self.sensor_data_at_triangulation = [] #the sensor data at the moment where we saved each of those corners
        self.current_gate = 0
        self.triangulated_gates= [None]*5#-np.ones((5,5,3)) # gates discovered by triangulation, dimensions : [gate_nb, point, coord]
        self.latest_triangulated_gate = None  # most recent triangulated gate in world coordinates
        #where gate_nb is 0 to 4, point nb is [top-left, top-right, bottom-right, bottom-left, center] and coord is [x,y,z]

        self.yaw_correction_gain = 10
        ######
        self.hover_z = 1.3
        self.search_yaw_rate = 0.25
        self.required_detection_frames = 3
        self.crawl_distance = 0.3 #small crawl forward when further away
        self.leap_distance = 0.6 #large leap when drone is closer than gate_close_distance_m
        self.approach_tolerance_m = 0.03
        self.gate_real_height_m = 0.4  # Real vertical height of the gate in meters
        self.gate_close_distance_m = 0.4  # Distance threshold to trigger close approach
        self.gate_close_warning = 0 #to prevent false positives, wait for 10 warnings of distance < gate_close_distance_m to trigger leap
        
    
        self.phase = "takeoff"
        self.search_target_yaw = None
        # self.detected_frames = 0
        self.approach_target_xyz = None
        # self.approach_target_z = None
        self.approach_target_yaw = None
        self.leap_target_xyz = None
        self.leap_yaw_target = None
        self.last_gate = None
        self.scan_zone_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)] #from zone 1 you observe zone, from zone 3 you observe zone 4, etc
        self.scan_zone_index = 0
        self.scan_speed_mps = 0.20
        self.scan_wp_tolerance_m = 0.12
        self.scan_entry_radius = 0.35
        self.scan_exit_margin = 0.15
        self.zone_start_point = None
        self.zone_end_point = None
        self.look_point = None
        self.latest_primary_gate_corners = None
        self.fly_through_offset_m = 0.4
        self.fly_through_wp_tolerance_m = 0.12
        self.fly_through_entry_point = None
        self.fly_through_exit_point = None
        self.observation_offset_m = 1
        self.observation_circle_radius_m = 0.5
        self.observation_wp_tolerance_m = 0.12
        self.observation_circle_points = None
        self.observation_circle_index = 0
        self.observation_circle_center = None
        self.observation_circle_normal = None
        self.race_trajectory_setpoints = None
        self.race_time_setpoints = None
        self.race_elapsed = 0.0
        self.race_grid_size = 0.25
        self.race_bounds = (0.0, 8.0, 0.0, 8.0, 0.0, 2.5)
        self.race_obstacles = []
        #####

         #values from Charbel's triangulation appendix
        self.camera_FOV = 1.5 #in [rad]
        self.focal_length = self.img_width / (2*np.tan(self.camera_FOV / 2.0)) #in [pixel]
        self.camera_position_relative_to_drone = [0.03, 0, 0.01] #in body frame 
        #other variables used for triangulation
        self.weight_sum = 0
        self.move_between_zones_speed = 1
        self.triangulation_improvement = np.inf
        self.stabilization_threshold = 0.05
        self.iterations_since_stabilization = 0
        self.nb_stable_iterations_to_move_on = 5
        self.min_distance_between_triangulations = 0.10  # Only triangulate if drone has moved this far (meters)
        self.last_triangulation_position = None  # Tracks drone position at last triangulation
        #variables used for vision
        self.px_margin = 3 #minimum 3 pixels between edge of gate and edge of screen to consider "fully visible"


    def get_latest_feed(self):
        with self.latest_feed_lock:
            if self.latest_feed is None:
                return None
            return self.latest_feed.copy()
        
    def find_center(self, P0,P1,P2,P3):
        '''Calculates the center of a rectangle projected in any perspective
        inputs : 4 corners P_i represented as list [x_i,y_i] of 2 pixel coordinates
        P0 and P2 are opposite corners and P1 and P3 too
        returns : (cx,cy) the pixel coordinates of the center point '''
        #we use the diagonals technique : draw a diagonal between opposite corners, parametrize these lines with L0_2(t) = P0 + t*(P2-P0)
        #and L1_3(t) = ..., write out system of equations for the center point L0_2(t_A) = L3_1(t_B) (where t_A and t_B are the points where
        #first and 2nd diagonal pass the centerpoint respectively), isolate t_B, then C = L3_1(t_B) 
        x0, y0 = P0
        x1, y1 = P1
        x2, y2 = P2
        x3, y3 = P3
        t_B = (y1 - y0 - x1 -x0) / ( (x2-x0)*(y2-y0) / ((x3-x1)*(y2-y0)-(y3-y1)*(x2-x0)) )
        cx = x1 + t_B * (x3-x1)
        cy = y1 + t_B * (y3-y1)
        return (cx,cy)
    
    def calculate_weight(self, gate_points, position1):
        d = np.linalg.norm(gate_points[4] - position1) #let's just take the distance to the center  (5th point of gate) of gate as metric
        # distance_weight = 1 / d**2 # 1/error = precision, if d decreases precision increases -> more weight
        # d_weight = min(distance_weight, 10) #cap the distance weight at 10 because it's improbable that the drone would actually be closer than 0.33m
        sigma_d = 1 #standard dev of distance 
        d_weight = np.exp((-d**2) / (2*sigma_d**2)) #gaussian weighting -> 1 if distance->0


        h_left = np.linalg.norm(gate_points[0] - gate_points[3]) #height of left side
        h_right = np.linalg.norm(gate_points[2] - gate_points[1])
        vh_left = np.abs(gate_points[0][2] - gate_points[3][2])
        vh_right = np.abs(gate_points[2][2] - gate_points[1][2])
        delta_l, delta_r = 0.4 - h_left, 0.4 - h_right
        sigma = 0.1  # standard deviation
        w_l = np.exp(-delta_l**2 / (2 * sigma**2)) #gaussian weighting
        w_r = np.exp(-delta_r**2 / (2 * sigma**2)) #weight -> 1 if delta->0
        h_weight = (w_l + w_r) / 2
        weight = (d_weight + h_weight) / 2
        # weight = d_weight
        # print(f"h_left {h_left:.3f} h_r = {h_right:.3f} vhl = {vh_left:.3f} vhr = {vh_right:.3f}")
        # print(f"tot = {weight:.3f}, d_w = {d_weight:.3f}, h_weight = {h_weight:.3f}, d = {d:.3f},, w_l = {w_l:.3f}, w_r ={w_r:.3f}")
        return max(weight, 0.0000001)


    def try_to_save_triangulation_data(self, sensor_data, corners):
        x = sensor_data['x_global']
        y = sensor_data['y_global']
        z = sensor_data['z_global']

        #if we're too close to last triangulated point, don't triangulate again (too similar results)
        if self.last_triangulation_position is not None: #check that this variable actually exists
            if np.linalg.norm(np.array([x,y,z]) - self.last_triangulation_position) < self.min_distance_between_triangulations:
                return False

        #if we're at a position to triangulate, see if there is gate in frame and choose the closest one
        primary_gate = self._select_primary_gate(corners) 
        if primary_gate is None: #if we cannot detect a gate which is fully in view
            return False
        
        self.latest_primary_gate_corners = np.array(primary_gate, dtype=float) #save for triangulation
        self.triangulation_list.append(self.latest_primary_gate_corners) 
        self.sensor_data_at_triangulation.append(sensor_data.copy()) #save position and orientation for triangulation
        self.last_triangulation_position = np.array([x,y,z]) #remember that we triangulated here
        # if len(self.triangulation_list) >= 2:
        #     if self.last_triangulation_position is None or np.linalg.norm(drone_pos - self.last_triangulation_position) >= self.min_distance_between_triangulations:
        #             self.triangulate_and_average(corners)
        # #             self.adjust_hover_height(z)
        # #             self.last_triangulation_position = np.array([x,y,z])
        # if 
        # self.triangulation_list.append(self.latest_primary_gate_corners)
        # self.sensor_data_at_triangulation.append(sensor_data.copy())

        return True

    

    def triangulate_and_average(self):
        '''This functions uses the last 2 triangulated positions to update the believed positions of the gate corners using
        inverse weighted averaging'''

        if len(self.triangulation_list) < 2: #need at least two positions to triangulate
            return None
        corners1, corners2 = self.triangulation_list[-1], self.triangulation_list[-2] #get the latest two sets of corners
        sd1, sd2 = self.sensor_data_at_triangulation[-1], self.sensor_data_at_triangulation[-2]
        position1 = [sd1["x_global"], sd1["y_global"], sd1["z_global"]]
        # position2 = [sd2["x_global"], sd2["y_global"], sd2["z_global"]]
        ordered1, ordered2 = self.sort_corners_by_angle(corners1,corners2) #sort them in the same order to be able to match them

        assert len(ordered1) == len(ordered2) == 4, f"error : Polygon with more than 4 corners in triangulate()"
        gate_points = [] #should be ordered top_left, top-right,bottom-right,bottom-left, center
        for i in range(4):
            gate_points.append(self.triangulate_one_point_between_two_frames(ordered1[i,:],sd1,ordered2[i,:],sd2))
        gate_points.append(np.mean(gate_points, axis=0)) #also add the center at the end #in 3D space the center point of rectangle is just the means of x,y and z 
        gate_points = np.array(gate_points)  #5 x 3 array [[TLx,TLy,TLz], [TRx,TRy,TRz], ..., [Cx,Cy,Cz]]
        self.latest_triangulated_gate = gate_points.copy()

        if self.which_zone(gate_points[4][0], gate_points[4][1]) != (2 + self.current_gate*2): #gate 0 is in zone 2, gate 1 is in zone 4, gate 2 is in zone 6, etc
        #    print(f"not in good zone (should be {(2 + self.current_gate*2)} is {self.which_zone(gate_points[4][0], gate_points[4][1])}) : {gate_points[4]}")
           return None
        
        weight = self.calculate_weight(gate_points, position1)
        # d = np.linalg.norm(gate_points[4] - position1) #let's just take the distance to the center  (5th point of gate) of gate as metric
        # distance_weight = 1 / d**2 # 1/error = precision, if d decreases precision increases -> more weight
        # weight = min(distance_weight, 10) #cap the distance weight at 10 because it's improbable that the drone would actually be closer than 0.33m

        if self.triangulated_gates[self.current_gate] is None: #if it's the first time we triangulate it -> save it
            self.triangulated_gates[self.current_gate] = gate_points
            self.triangulation_iteration = 1
            print(f"first triangulation of gate {self.current_gate} : {gate_points}")
            self.weight_sum += weight

        
        else: 
            last_avg_triangulation = self.triangulated_gates[self.current_gate] 
            previous_num = last_avg_triangulation * self.weight_sum #old_numerator = T1 + T2 + ... + Ti = avg_gate * (w1 + w2 + ... + wi)
            self.weight_sum += weight #add the weight wi+1 of this new triangulation to the weight sum
            new_avg = (previous_num + gate_points*weight) / self.weight_sum #new avg = (old_numerator + Ti+1) / (w1 + w2 + ... +  wi + wi+1)
            self.triangulated_gates[self.current_gate] = new_avg

            #calculate how much this new triangulation has improved the certainty on the position
            triangulation_improvement = abs(np.linalg.norm(last_avg_triangulation[4] - new_avg[4]))
            #this way we make our triangulation measurement better at every step
              
            if triangulation_improvement <= self.stabilization_threshold :
                self.iterations_since_stabilization += 1 #count how many iterations in a row of triangulation don't improve significantly position
            else :
                self.iterations_since_stabilization = 0 #reset
            self.triangulation_iteration += 1
            # print(f"new ({self.triangulation_iteration}) refinement of gate {self.current_gate} : weight {weight},  improvement = {triangulation_improvement:.3f}")

        return gate_points

    def which_zone(self, x, y):
        """
        Given a point (x, y) in the 8x8 square (origin at bottom-right,
        x-axis forward, y-axis left), returns the zone number (0-11).

        Zone 1: centered on +x direction, spanning [-15°, +15°)
        Zones increase counter-clockwise.

        Returns -1 if the point is exactly at the center (4, 4).
        """
        if not(0 <= x <= 8) or not(0 <= y <= 8):
            print("zone error : point not in arena")
            return -1
        if x == 4.0 and y == 4.0:
            print("zone error : point exactly in the middle (undefined)")
            return -2

        cx, cy = 4.0, 4.0

        dx = x - cx
        dy = y - cy

        # angle in degrees, in [-180, 180]
        angle_deg = math.degrees(math.atan2(dy, dx))

        #if x axis points up, zone 6 (top zone) is -15 to +15 and takeoff zone (0)
        #is offset by 180 so 165 to 195
        # -> offset by 180 + 15 degrees so Zone 0 maps to [0, 30[ after the shift
        angle_shifted = (angle_deg + 195.0) % 360.0

        zone = int(angle_shifted // 30)
        return zone



    def sort_corners_by_angle(self, corners1, corners2):
        '''Takes two lists of 4 points (representing the 4 corners of a gate on two different but close images)
        and orders them both in the same order top-left, top-right,bottom-right,bottom-left
        inputs : corners1 and corners 2 (each is a list of four points)
        returns : two lists of four points ordered similarly'''

        #to do this we use the angle of the corner with respect to the centroid : since the drone should keep its roll 0,
        #the angles should stay consistent between frames
        # matched_corners = np.zeros((4,2))
        # print(f"unordered {corners1} and {corners2}")
        corners1, corners2 = np.array(corners1).squeeze(), np.array(corners2).squeeze() #transform in array and squeeze unecessary dimensions
        cx1, cy1 = np.mean(corners1,axis=0) #get centroid in x and y
        angles1 = np.arctan2(corners1[:,1] - cy1, corners1[:,0] - cx1)
        ordered1 = corners1[np.argsort(angles1)]

        cx2, cy2 = np.mean(corners1,axis=0) #get centroid x and y
        angles2 = np.arctan2(corners2[:,1] - cy2, corners2[:,0] - cx2)
        ordered2 = corners2[np.argsort(angles2)]
        # print("ordered corners")
        # print(ordered1)
        # print(ordered2)
        return ordered1, ordered2



    def triangulate_one_point_between_two_frames(self, pixel_coords1, data1, pixel_coords2, data2):
        ''' Triangulate the position of a point in world (=inertial) frame based on the pixel coordinates of this
        point taken from two different camera positions
        inputs : 
        pixel_coords1 = [px1, py1] x and y pixel coordinates of point in first image
        data1 = the sensor data at the time when image 1 was taken
        pixel_coords2 = [px2, py2]
        data2 = the sensor data at the time when image 2 was taken
        returns : [x,y,z] position of triangulated point in world frame '''
        R_body1_to_world = R.from_quat([data1["q_x"], data1["q_y"], data1["q_z"], data1["q_w"]]).as_matrix()  # Rotation from body to inertial frame
        R_cam1_to_world = R_body1_to_world @ self.cam_to_body
        R_body2_to_world = R.from_quat([data2["q_x"], data2["q_y"], data2["q_z"], data2["q_w"]]).as_matrix()  # Rotation from body to world frame
        R_cam2_to_world = R_body2_to_world @ self.cam_to_body

        P1 = np.array([data1["x_global"], data1["y_global"], data1["z_global"]]) #P1 corresponds to P in the triangulation appendix made by Charbel
        P1 += R_body1_to_world @ self.camera_position_relative_to_drone #add the offset between body and camera
        P2 = np.array([data2["x_global"], data2["y_global"], data2["z_global"]]) #P2 is Q
        P2 += R_body2_to_world @ self.camera_position_relative_to_drone 

        #v1 and v2 obtained from moving origin to center of image and the z component is focal length
        v1 = np.array([pixel_coords1[0] - self.img_width/2 , pixel_coords1[1] - self.img_width/2, self.focal_length]) 
        v2 = np.array([pixel_coords2[0] - self.img_width/2 , pixel_coords2[1] - self.img_width/2, self.focal_length]) #should be img_height/2 but it's a square img so it's the same

        r1 = R_cam1_to_world @ v1 #the line going from position 1 to pixel 1 in space is parametrized as : P1 + r1 * lambda
        r2 = R_cam2_to_world @ v2 #the line going from position 2 to pixel 2 in space is P2 + r2 * mu

        #calculate the time where the lines are the closest to each other, thus the closest to the real position in space of the point we want to triangulate H
        #to do this we use the pseudo inverse
        A = np.array([[r1[0], -r2[0]], 
                      [r1[1], -r2[1]], 
                      [r1[2], -r2[2]]])
        lambda_h, mu_h = np.linalg.pinv(A) @ (P2 - P1)

        #the closest point on each line is respectively H1 = P1 + r1*lambda_h and H2 = P2 + r2*mu_h
        #thus we can approximate H as (H1 + H2)/2
        H = ((P1 + r1*lambda_h) + (P2 + r2*mu_h)) / 2
        
        return H #this is the triangulated point [Hx, Hy, Hz]
    
    def next_gate(self):
        ''' Resets all attributes necessary for triangulation of one gate'''
        # self.triangulation_improvement = np.inf
        self.last_list_length = 0
        self.hover_z = 1.3 #reset to default height
        self.iterations_since_stabilization = 0
        self.weight_sum = 0
        self.triangulation_list = []
        self.sensor_data_at_triangulation =[]
        self.latest_triangulated_gate = None
        self.last_triangulation_position = None  # Reset position tracker for new gate
        print(f"current_gate {self.current_gate}")
        if self.current_gate == 4:
            self.phase = "race"
            print(f"finished first lap, {self.phase} time")

        else:
            self.current_gate += 1


    def _project_world_point_to_camera(self, world_point, sensor_data):
        """Project a 3D world point to 2D camera pixel coordinates.
        
        world_point: np.array shape (3,) in world frame
        sensor_data: dict with q_x, q_y, q_z, q_w, x_global, y_global, z_global
        returns: (px, py) tuple or None if point is behind camera
        """
        # Get body-to-world rotation
        R_body_to_world = R.from_quat([sensor_data["q_x"], sensor_data["q_y"], 
                                        sensor_data["q_z"], sensor_data["q_w"]]).as_matrix()
        R_cam_to_world = R_body_to_world @ self.cam_to_body
        
        # Get camera position in world frame
        drone_pos = np.array([sensor_data["x_global"], sensor_data["y_global"], 
                              sensor_data["z_global"]], dtype=float)
        cam_pos = drone_pos + R_body_to_world @ self.camera_position_relative_to_drone
        
        # World point relative to camera origin
        point_rel_cam = world_point - cam_pos
        
        # Transform to camera frame
        R_world_to_cam = R_cam_to_world.T
        point_cam = R_world_to_cam @ point_rel_cam
        
        # Check if point is in front of camera (z > 0)
        if point_cam[2] <= 0:
            return None
        
        # Project to camera image plane using pinhole model
        px = self.focal_length * point_cam[0] / point_cam[2] + self.img_width / 2.0
        py = self.focal_length * point_cam[1] / point_cam[2] + self.img_width / 2.0
        
        # Check if within image bounds
        if 0 <= px < self.img_width and 0 <= py < self.img_width:
            return (int(px), int(py))
        return None

    def vision(self, camera_data, sensor_data=None):

        # cv2.imshow("test",camera_data)
        # cv2.waitKey(1)

        # gate_image = cv2.imread("/home/julien/Documents/EPFL_Master/Aerial/micro-502/gates.png")
        gate_image = camera_data
        if gate_image.ndim == 3 and gate_image.shape[2] == 4: #confirm it's a RGBA + A image
            gate_image = cv2.cvtColor(gate_image, cv2.COLOR_BGRA2BGR) #convert to simple RGB
        # --- 1. Pink Color Mask (HSV) ---
        hsv = cv2.cvtColor(gate_image, cv2.COLOR_BGR2HSV) #convert to HSV color space (more robust)

        lower_pink = np.array([140, 50, 50]) #pink range
        upper_pink = np.array([175, 255, 255])

        mask = cv2.inRange(hsv, lower_pink, upper_pink) #threshold to just keep pixels in pink range

        #morphological opening and closing like in image processing class to remove noise and artefacts (maybe not  useful ?)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  

        masked_image = cv2.bitwise_and(gate_image, gate_image, mask=mask) #apply mask to image
        gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY) #convert to grayscale 
        edges = cv2.Canny(gray_masked, threshold1=50, threshold2=150) #canny edge detection

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 8. List to store the corners of the obstacles
        obstacle_corners = []

        # 9. Approximate the contours as polygons
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Approximation precision
            approx = cv2.approxPolyDP(contour, epsilon, True)
            #get the convex hull of the approximated contour to get rid of concavities and get a more regular shape (we are looking for quadrilaterals)
            hull = cv2.convexHull(approx)
            # Add the 4 points of the hull (corners of the quadrilateral)
            if len(hull) == 4: #only consider contours that are approximated to quadrilaterals
                points = hull.reshape(-1, 2).tolist() 
                for (x,y) in points: #only keep if the frame is fully in frame (edge of gate doesn't touch edge of image)
                    if ((self.px_margin < x < self.img_width - self.px_margin) and 
                        (self.px_margin < x < self.img_width - self.px_margin)):
                        obstacle_corners.append(points)

        # 10. Visualize the polygons with colored edges and circles at corners
        img_with_polygons = gate_image.copy()

        # # Assign a unique color for each polygon
        # for i, polygon in enumerate(obstacle_corners):
        #     # Random color for each polygon (BGR format)
        #     color = np.random.randint(0, 256, 3).tolist()

        #     # Draw the corners as red circles
        #     # for (x, y) in polygon:
        #     #     cv2.circle(img_with_polygons, (x, y), 5, (0, 0, 255), -1)  # Red circles at corners

        #     # Draw the polygon edges
            # polygon_array = np.array(polygon, dtype=np.int32)
            # cv2.drawContours(img_with_polygons, [polygon_array], -1, color, 2)  # Polygon edges with random color


        # cv2.imshow("Original", gate_image)
        # cv2.imshow("Pink Mask", mask)
        # cv2.imshow("Masked Image", masked_image)
        # cv2.imshow("Edges", edges)
        # cv2.imshow("img with polygones", img_with_polygons)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Draw triangulated gate points if available
        if sensor_data is not None and self.triangulated_gates[self.current_gate] is not None:
            gate_points_3d = self.triangulated_gates[self.current_gate]  # shape (5,3)
            
            # Project each gate point to camera frame
            projected_pts = []
            for i, pt_3d in enumerate(gate_points_3d):
                pix = self._project_world_point_to_camera(pt_3d, sensor_data)
                if pix is not None:
                    projected_pts.append(pix)
                    # Draw corner points (0-3) in green, center (4) in blue
                    if i < 4:
                        cv2.circle(img_with_polygons, pix, 8, (0, 255, 0), -1)  # Green corners
                    else:
                        cv2.circle(img_with_polygons, pix, 10, (255, 0, 0), -1)  # Blue center
            if len(projected_pts) >= 4:
                for i in range(4):
                    cv2.line(img_with_polygons, projected_pts[i], projected_pts[(i+1)%4], (0, 255, 255), 2)

        # Draw the latest triangulated frame in red for comparison
        if sensor_data is not None and self.latest_triangulated_gate is not None:
            latest_gate_points_3d = self.latest_triangulated_gate
            latest_projected_pts = []
            for i, pt_3d in enumerate(latest_gate_points_3d):
                pix = self._project_world_point_to_camera(pt_3d, sensor_data)
                if pix is not None:
                    latest_projected_pts.append(pix)
                    if i < 4:
                        cv2.circle(img_with_polygons, pix, 6, (0, 0, 255), -1)  # Red corners
                    else:
                        cv2.circle(img_with_polygons, pix, 8, (0, 0, 180), -1)  # Darker red center

            if len(latest_projected_pts) >= 4:
                for i in range(4):
                    cv2.line(img_with_polygons, latest_projected_pts[i], latest_projected_pts[(i+1)%4], (0, 0, 255), 1)
        
        with self.latest_feed_lock:
            self.latest_feed = img_with_polygons.copy()

        self.latest_obstafles_corners = obstacle_corners

        return obstacle_corners

################33
    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _select_primary_gate(self, obstacle_corners):
        if len(obstacle_corners) == 0:
            return None
        
        # # print(f"obstacle corners length {len(obstacle_corners)} : {obstacle_corners}")
        # if len(obstacle_corners) == 1: #if there's only one gate in view
        #     return np.array(obstacle_corners, dtype=np.float32)

        min_distance = 0.0
        closest_polygon = None
        for polygon in obstacle_corners:
            pts = np.array(polygon, dtype=np.float32)
            # calculate longest vertical side by checking all edges
            d = self._distance_to_gate(pts)
            # for i in range(len(pts)):
            #     p1 = pts[i]
            #     p2 = pts[(i + 1) % len(pts)]
            #     vertical_distance = abs(p2[1] - p1[1])
            #     if vertical_distance > max_vertical_distance:
            #         max_vertical_distance = vertical_distance
            
            if d > min_distance:
                min_distance = d
                closest_polygon = pts

        return closest_polygon

    def _gate_center(self, polygon):
        """Calculate the center of a polygon using image moments.
        
        polygon: numpy array of polygon vertices
        returns: (center_x, center_y) tuple of pixel coordinates
        """
        moments = cv2.moments(polygon)
        if abs(moments["m00"]) > 1e-6:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            return center_x, center_y
        else:
            # Fallback to mean if moments calculation fails
            return float(np.mean(polygon[:, 0])), float(np.mean(polygon[:, 1]))

    # def _gate_orientation(self, polygon):
    #     """Calculate the orientation angle of the gate polygon.
        
    #     The orientation is the angle of the major axis of the polygon.
    #     For a gate perpendicular to the drone, this should be ~90 degrees (vertical).
        
    #     polygon: numpy array of polygon vertices
    #     returns: angle in radians (-pi/2 to pi/2) representing the gate's tilt
    #     """
    #     moments = cv2.moments(polygon)
    #     if abs(moments["m00"]) < 1e-6:
    #         return 0.0
        
    #     # Central moments
    #     mu20 = moments["mu20"] / moments["m00"]
    #     mu02 = moments["mu02"] / moments["m00"]
    #     mu11 = moments["mu11"] / moments["m00"]
        
    #     # Calculate orientation angle using principal axis formula
    #     # angle = 0.5 * atan2(2*mu11, mu20 - mu02)
    #     # This gives the angle of minimum inertia axis
    #     angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        
    #     return angle

    # def _bearing_from_pixel_x(self, center_x):
    #     x_norm = (center_x - (self.img_width / 2.0)) / (self.img_width / 2.0)
    #     x_norm = np.clip(x_norm, -1.0, 1.0)
    #     return -x_norm * (self.camera_FOV / 2.0)

    # def _pitch_from_pixel_y(self, center_y):
    #     """Convert pixel y-coordinate to pitch angle (vertical angle).
        
    #     center_y: y pixel coordinate of gate center (0=top, 150=center, 299=bottom)
    #     returns: pitch angle in radians (positive=look up, negative=look down)
    #     """
    #     y_norm = (center_y - (self.img_width / 2.0)) / (self.img_width / 2.0)
    #     y_norm = np.clip(y_norm, -1.0, 1.0)
    #     # Positive y_norm (gate lower in image) -> negative pitch (look down, descend)
    #     return -y_norm * (self.camera_FOV / 2.0)
    
    def _distance_to_gate(self, polygon):
        """Estimate distance to the gate based on its vertical edge height.
        
        Uses the pinhole camera model: distance = (real_height * focal_length) / pixel_height
        The gate is known to have a vertical height of 0.4m.
        
        polygon: numpy array of polygon vertices (quadrilateral)
        returns: estimated distance in meters
        """
        pts = np.array(polygon, dtype=np.float32).squeeze() #convert to np array to be sure, but squeeze to prevent  unecessary dimensions
        # Find the longest vertical edge (the one with maximum y-difference)
        max_vertical_pixel_height = 0.0
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            vertical_distance = abs(p2[1] - p1[1])
            if vertical_distance > max_vertical_pixel_height:
                max_vertical_pixel_height = vertical_distance
        
        # Avoid division by zero
        if max_vertical_pixel_height < 1.0:
            return float('inf')
        
        # Use pinhole camera model: distance = (real_height * focal_length) / pixel_height
        distance = (self.gate_real_height_m * self.focal_length) / max_vertical_pixel_height
        
        return distance

    def _gate_normality_weight(self, polygon):
        """Estimate how fronto-parallel the gate looks in the image.

        Returns a value in [0, 1] where 1 means opposite edges are nearly
        parallel and similar in length.
        """
        pts = np.array(polygon, dtype=np.float32).squeeze()
        if pts.shape[0] != 4:
            return 0.0

        centroid = np.mean(pts, axis=0)
        order = np.argsort(np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0]))
        ordered = pts[order]

        edges = [ordered[(i + 1) % 4] - ordered[i] for i in range(4)]
        lengths = [np.linalg.norm(e) for e in edges]
        if min(lengths) < 1e-6:
            return 0.0

        def parallel_score(v1, v2):
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return 0.0
            return float(np.clip(abs(np.dot(v1, v2)) / (n1 * n2), 0.0, 1.0))

        def length_similarity(l1, l2):
            return 1.0 - abs(l1 - l2) / (l1 + l2 + 1e-6)

        opposite_parallel = 0.5 * (parallel_score(edges[0], edges[2]) + parallel_score(edges[1], edges[3]))
        opposite_length = 0.5 * (length_similarity(lengths[0], lengths[2]) + length_similarity(lengths[1], lengths[3]))

        normality = 0.75 * opposite_parallel + 0.25 * opposite_length
        return float(np.clip(normality, 0.0, 1.0))

    # def _heading_to_gate_world(self, center_x, yaw, gate):
    #     bearing = self._bearing_from_pixel_x(center_x)
    #     gate_orientation = self._gate_orientation(gate)
    #     d = self._distance_to_gate(gate)
    #     yaw_correction = self._yaw_correction_from_gate_orientation(gate_orientation, d)
    #     return self._wrap_angle(yaw + bearing + yaw_correction)
    
    # def go_towards_gate(self, x, y, z, yaw, gate, mvt_distance):
    #     center_x, center_y = self._gate_center(gate)
    #     pitch = self._pitch_from_pixel_y(center_y) #calculate height adjustement (based on y-axis in camera frame)
    #     heading_world = self._heading_to_gate_world(center_x, yaw, gate) #calculate yaw adjustement based 

    #     target_x = x + mvt_distance * np.cos(heading_world)
    #     target_y = y + mvt_distance * np.sin(heading_world)
    #     target_z = z + self._height_adjustment_from_pitch(pitch, mvt_distance)
    #     target_z = np.clip(target_z, 0.5, 2.2)  # safety bounds, gate center is necessarily between [.5 and 2.2]

    #     target_yaw = self._wrap_angle(heading_world)
    #     return [target_x, target_y, target_z, target_yaw]
    
    # def _height_adjustment_from_pitch(self, pitch, distance):
    #     """Convert pitch angle and forward distance to vertical distance to move.
        
    #     pitch: pitch angle in radians (negative=descend, positive=ascend)
    #     distance: forward distance to gate (meters)
    #     returns: vertical distance to move (positive=up, negative=down)
    #     """
    #     return distance * np.tan(pitch)

    # def _yaw_correction_from_gate_orientation(self, gate_orientation, distance):
    #     """Calculate yaw correction to align the drone perpendicular to the gate.
        
    #     The gate orientation is computed in image space. A vertical gate (orientation ~0)
    #     means the drone is already face-on. A tilted gate means we need to rotate.
        
    #     gate_orientation: angle in radians from _gate_orientation()
    #     returns: yaw correction angle in radians
    #     """
    #     # The gate orientation directly tells us how much to rotate
    #     # Positive angle = gate tilted CCW in image -> rotate CW (negative yaw)
    #     # Negative angle = gate tilted CW in image -> rotate CCW (positive yaw)
    #     if distance < self.gate_close_distance_m: #if we're too close, don't add a yaw correction
    #         return 0
    #     correction = -gate_orientation /(self.yaw_correction_gain * distance) #scale correction with distance
    #     correction = np.clip(correction, a_min=-gate_orientation/3, a_max=gate_orientation/3)
    #     return correction

    def _zone_center_angle(self, zone):
        """Return the center angle (rad) of a zone in the same convention as which_zone()."""
        angle_deg = zone * 30.0 - 180.0
        # print(f"_zone_center_angle({zone}) = {angle_deg}° = {np.deg2rad(angle_deg):.4f} rad")
        return np.deg2rad(angle_deg)

    def _ray_to_square_edge(self, direction):
        """Intersect ray from arena center with square border [0,8]x[0,8]."""
        center = np.array([4.0, 4.0], dtype=float)
        dx, dy = float(direction[0]), float(direction[1])
        t_candidates = []

        if dx > 1e-9:
            t_candidates.append((8.0 - center[0]) / dx)
        elif dx < -1e-9:
            t_candidates.append((0.0 - center[0]) / dx)

        if dy > 1e-9:
            t_candidates.append((8.0 - center[1]) / dy)
        elif dy < -1e-9:
            t_candidates.append((0.0 - center[1]) / dy)

        t_positive = [t for t in t_candidates if t > 0]
        if len(t_positive) == 0:
            return center.copy()

        t_edge = min(t_positive)
        return center + t_edge * np.array([dx, dy])

    def _zone_centerline_points(self, zone):
        """Return start, midpoint, and end points of a zone centerline segment."""
        center = np.array([4.0, 4.0], dtype=float)
        angle = self._zone_center_angle(zone)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        edge = self._ray_to_square_edge(direction)
        radius_to_edge = np.linalg.norm(edge - center)

        start_radius = min(self.scan_entry_radius, max(0.0, radius_to_edge - 0.05))
        end_radius = max(start_radius + 0.05, radius_to_edge - self.scan_exit_margin)
        midpoint_radius = 0.5 * radius_to_edge

        start = center + start_radius * direction
        midpoint = center + midpoint_radius * direction
        end = center + end_radius * direction
        
        angle_rad = np.rad2deg(angle)
        # print(f"  ZCP zone={zone}: angle={angle_rad:.1f}°, dir={direction}, edge={edge}, r2e={radius_to_edge:.2f}")
        
        return start, midpoint, end

    def _move_towards(self, current_xy, target_xy, step_size):
        delta = target_xy - current_xy
        dist = np.linalg.norm(delta)
        if dist <= 1e-9:
            return target_xy.copy(), dist
        if dist <= step_size:
            return target_xy.copy(), dist
        return current_xy + (step_size / dist) * delta, dist

    def _compute_gate_fly_through_points(self, gate_points, drone_position):
        """Return two waypoints around gate center along gate-plane normal.

        gate_points: np.array shape (5,3) [TR, BR, BL, TL, C]
        drone_position: np.array shape (3,)
        """
        p0 = gate_points[0]
        p1 = gate_points[1]
        p3 = gate_points[3]
        center = gate_points[4]

        in_plane_a = p1 - p0
        in_plane_b = p3 - p0
        normal = np.cross(in_plane_a, in_plane_b)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None, None
        normal = normal / norm

        # pick the normal sign so entry point is on current drone side
        if np.dot(drone_position - center, normal) < 0:
            normal = -normal

        entry = center + self.fly_through_offset_m * normal
        exit = center - self.fly_through_offset_m * normal
        return entry, exit

    def _compute_gate_observation_circle(self, gate_points, drone_position, num_points=12):
        """Return a list of waypoints on a circle parallel to the gate plane.

        The circle is centered on the gate center shifted by observation_offset_m
        along the gate normal, and has radius observation_circle_radius_m.
        """
        p0 = gate_points[0]
        p1 = gate_points[1]
        p3 = gate_points[3]
        center = gate_points[4]

        in_plane_a = p1 - p0
        in_plane_b = p3 - p0
        normal = np.cross(in_plane_a, in_plane_b)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None, None, None
        normal = normal / norm

        # Make the normal point toward the current drone side of the gate
        if np.dot(drone_position - center, normal) < 0:
            normal = -normal

        # Build an orthonormal basis spanning the gate plane
        u = in_plane_a / (np.linalg.norm(in_plane_a) + 1e-9)
        u = u - np.dot(u, normal) * normal
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-6:
            return None, None, None
        u = u / u_norm
        v = np.cross(normal, u)
        v = v / (np.linalg.norm(v) + 1e-9)

        circle_center = center + self.observation_offset_m * normal
        angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
        circle_points = [circle_center + self.observation_circle_radius_m * (
            np.cos(a) * u + np.sin(a) * v
        ) for a in angles]
        return np.array(circle_points), circle_center, normal

    def _current_gate_center_xy(self):
        gate_points = self.triangulated_gates[self.current_gate]
        if gate_points is None:
            return None
        return np.array(gate_points[4][:2], dtype=float)

    def _prepare_scan_segment(self):
        if self.scan_zone_index >= len(self.scan_zone_pairs):
            self.zone_start_point = None
            self.zone_end_point = None
            self.look_point = None
            return

        scan_zone, look_zone = self.scan_zone_pairs[self.scan_zone_index]
        start, _, end = self._zone_centerline_points(scan_zone)
        _, look_midpoint, _ = self._zone_centerline_points(look_zone)

        self.zone_start_point = start
        self.zone_end_point = end
        self.look_point = look_midpoint
        # self.detected_frames = 0
        
        angle_deg = scan_zone * 30.0 - 180.0
        print(f"ZONE_PREP: scan_zone={scan_zone}, angle_deg={angle_deg:.1f}, start={start}, end={end}, look={look_midpoint}")

    def _advance_to_next_zone(self):
        self.scan_zone_index += 1
        if self.scan_zone_index >= len(self.scan_zone_pairs):
            return
        self._prepare_scan_segment()
        self.phase = "move_to_zone_start"

    def _start_observation_phase(self, gate_points, drone_position):
        circle_points, circle_center, normal = self._compute_gate_observation_circle(gate_points, drone_position)
        if circle_points is None:
            return False
        self.observation_circle_points = circle_points
        self.observation_circle_center = circle_center
        self.observation_circle_normal = normal
        self.observation_circle_index = 0
        self.phase = "observation"
        print(f"observation phase: center={circle_center}, normal={normal}, points={len(circle_points)}")
        return True

    def _build_race_trajectory(self, sensor_data):
        """Build one concatenated motion-planned trajectory through gates 0-4 twice."""
        if any(self.triangulated_gates[i] is None for i in range(5)):
            print("race phase: missing triangulated gate data")
            return False

        start = (
            float(sensor_data["x_global"]),
            float(sensor_data["y_global"]),
            float(sensor_data["z_global"]),
        )

        all_setpoints = []
        all_timepoints = []
        current_start = start
        time_offset = 0.0

        for gate_idx in [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]: #pass trough each gate 2 times
            gate_points = self.triangulated_gates[gate_idx]
            goal = tuple(float(v) for v in gate_points[4])

            try:
                planner = _RaceMotionPlanner3D(
                    current_start,
                    self.race_obstacles,
                    self.race_bounds,
                    self.race_grid_size,
                    goal,
                )
            except Exception as e:
                print(f"race phase: motion planner failed for gate {gate_idx}: {e}")
                return False

            segment_setpoints = np.asarray(planner.trajectory_setpoints, dtype=float)
            segment_timepoints = np.asarray(planner.time_setpoints, dtype=float)

            if segment_setpoints.size == 0 or segment_timepoints.size == 0:
                print(f"race phase: empty trajectory for gate {gate_idx}")
                return False

            if len(all_setpoints) > 0:
                segment_setpoints = segment_setpoints[1:]
                segment_timepoints = segment_timepoints[1:]

            segment_timepoints = segment_timepoints + time_offset
            time_offset = float(segment_timepoints[-1])

            all_setpoints.append(segment_setpoints)
            all_timepoints.append(segment_timepoints)
            current_start = goal

        self.race_trajectory_setpoints = np.vstack(all_setpoints)
        self.race_time_setpoints = np.concatenate(all_timepoints)
        self.race_elapsed = 0.0
        print(f"race phase: built trajectory with {len(self.race_trajectory_setpoints)} setpoints")
        return True

    def adjust_hover_height(self, current_z):
        avg_gate = self.triangulated_gates[self.current_gate]
        if avg_gate is not None:
            target_z = avg_gate[4][2] #use z coord of gate center as target
            correction = (target_z - current_z)*0.1
            self.hover_z = current_z + correction
            # print(f"current z {current_z}, target_z = {target_z}, correction = {correction}")
    


    def state_machine(self, sensor_data, corners, dt):
        x = sensor_data['x_global']
        y = sensor_data['y_global']
        z = sensor_data['z_global']
        yaw = sensor_data['yaw']

        if self.phase == "takeoff":
            if z >= self.hover_z - 0.1:
                self._prepare_scan_segment()
                if self.zone_start_point is None:
                    self.phase = "hold"
                else:
                    self.phase = "move_to_zone_start"
                    print(f"move to zone start")

            return [x, y, self.hover_z, yaw]

        if self.phase == "move_to_zone_start":
            scan_zone = self.scan_zone_pairs[self.scan_zone_index][0] if self.scan_zone_index < len(self.scan_zone_pairs) else -1
            if self.zone_start_point is None or self.zone_end_point is None or self.look_point is None:
                self._prepare_scan_segment()
                if self.zone_start_point is None:
                    self.phase = "hold"
                    return [x, y, self.hover_z, yaw]

            current_xy = np.array([x, y], dtype=float)
            step = self.move_between_zones_speed
            next_xy, dist_to_start = self._move_towards(current_xy, self.zone_start_point, step)
            target_yaw = self._wrap_angle(np.arctan2(self.look_point[1] - y, self.look_point[0] - x))
            # print(f"  pos=({x:.2f},{y:.2f}) cmd=({next_xy[0]:.2f},{next_xy[1]:.2f}) dist={dist_to_start:.3f} step={step:.4f}")

            if dist_to_start <= self.scan_wp_tolerance_m:
                self.phase = "scan_zone"
                print("zone scan")


            return [float(next_xy[0]), float(next_xy[1]), self.hover_z, target_yaw]

        if self.phase in ("scan_zone", "scan_zone_return"):
            current_xy = np.array([x, y], dtype=float)
            step = self.scan_speed_mps

            if self.phase == "scan_zone":
                target_xy = self.zone_end_point
                look_xy = self.look_point
            else:
                target_xy = self.zone_start_point
                look_xy = self._current_gate_center_xy()
                if look_xy is None:
                    look_xy = self.look_point

            next_xy, dist_to_target = self._move_towards(current_xy, target_xy, step)
            target_yaw = self._wrap_angle(np.arctan2(look_xy[1] - y, look_xy[0] - x))
            if self.try_to_save_triangulation_data(sensor_data, corners):
                self.triangulate_and_average()
                self.adjust_hover_height(z)

            if self.iterations_since_stabilization > self.nb_stable_iterations_to_move_on:
                gate_points = self.triangulated_gates[self.current_gate]
                print(f"good enough triangulation after second pass, gate at {gate_points}")
                if gate_points is not None:
                    drone_pos = np.array([x, y, z], dtype=float)
                    if self._start_observation_phase(gate_points, drone_pos):
                        obs_target = self.observation_circle_points[self.observation_circle_index]
                        obs_yaw = self._wrap_angle(np.arctan2(gate_points[4][1] - y, gate_points[4][0] - x))
                        return [float(obs_target[0]), float(obs_target[1]), float(obs_target[2]), obs_yaw]

                print("observation setup failed, advancing")
                self.next_gate()
                self._advance_to_next_zone()
                return [float(next_xy[0]), float(next_xy[1]), self.hover_z, target_yaw]

            if dist_to_target <= self.scan_wp_tolerance_m:
                if self.phase == "scan_zone":
                    print("reached end of scan zone, starting second pass")
                    self.phase = "scan_zone_return"
                else:
                    print("reached start of second pass, advancing")
                    gate_points = self.triangulated_gates[self.current_gate]
                    if gate_points is not None and self._start_observation_phase(gate_points, np.array([x, y, z], dtype=float)):
                        pass
                    else:
                        self.next_gate()
                        self._advance_to_next_zone()

            return [float(next_xy[0]), float(next_xy[1]), self.hover_z, target_yaw]

        if self.phase == "observation":
            if self.observation_circle_points is None or len(self.observation_circle_points) == 0:
                print("missing observation circle, advancing")
                self.next_gate()
                self._advance_to_next_zone()
                return [x, y, self.hover_z, yaw]
            
            if self.try_to_save_triangulation_data(sensor_data, corners):
                self.triangulate_and_average()

            target = self.observation_circle_points[self.observation_circle_index]
            gate_points = self.triangulated_gates[self.current_gate]
            gate_center = gate_points[4] if gate_points is not None else target
            target_yaw = self._wrap_angle(np.arctan2(gate_center[1] - y, gate_center[0] - x))

            dist_to_target = np.linalg.norm(np.array([x, y, z], dtype=float) - target)
            if dist_to_target <= self.observation_wp_tolerance_m:
                self.observation_circle_index += 1
                if self.observation_circle_index >= len(self.observation_circle_points):
                    print("observation circle complete, preparing fly-through")
                    drone_pos = np.array([x, y, z], dtype=float)
                    if gate_points is not None:
                        entry, exit = self._compute_gate_fly_through_points(gate_points, drone_pos)
                        if entry is not None and exit is not None:
                            self.fly_through_entry_point = entry
                            self.fly_through_exit_point = exit
                            self.phase = "fly_through_entry"
                            print(f"fly through gate {self.current_gate}: entry={entry}, exit={exit}")
                            return [float(entry[0]), float(entry[1]), float(entry[2]), target_yaw]

                    print("fly-through setup failed after observation, advancing")
                    self.next_gate()
                    self._advance_to_next_zone()
                    return [x, y, self.hover_z, target_yaw]

                target = self.observation_circle_points[self.observation_circle_index]

            return [float(target[0]), float(target[1]), float(target[2]), target_yaw]

        if self.phase == "fly_through_entry":
            if self.fly_through_entry_point is None or self.fly_through_exit_point is None:
                print("missing fly-through waypoints, advancing")
                self.next_gate()
                self._advance_to_next_zone()
                return [x, y, self.hover_z, yaw]

            target = self.fly_through_entry_point
            dist_to_entry = np.linalg.norm(np.array([x, y, z], dtype=float) - target)
            target_yaw = self._wrap_angle(np.arctan2(self.fly_through_exit_point[1] - y,
                                                     self.fly_through_exit_point[0] - x))

            if dist_to_entry <= self.fly_through_wp_tolerance_m:
                self.phase = "fly_through_exit"
                print("fly-through: reached entry, heading to exit")

            return [float(target[0]), float(target[1]), float(target[2]), target_yaw]

        if self.phase == "fly_through_exit":
            if self.fly_through_exit_point is None:
                print("missing exit waypoint, advancing")
                self.next_gate()
                self._advance_to_next_zone()
                return [x, y, self.hover_z, yaw]

            target = self.fly_through_exit_point
            dist_to_exit = np.linalg.norm(np.array([x, y, z], dtype=float) - target)
            target_yaw = self._wrap_angle(np.arctan2(target[1] - y, target[0] - x))

            if dist_to_exit <= self.fly_through_wp_tolerance_m:
                print("fly-through: gate passed")
                self.fly_through_entry_point = None
                self.fly_through_exit_point = None
                self.next_gate()
                self._advance_to_next_zone()

            return [float(target[0]), float(target[1]), float(target[2]), target_yaw]
        
        if self.phase == "race" :
            print("race phase")
            print(f"{len(self.triangulated_gates)} triangulated gates : {self.triangulated_gates} ")
            if self.race_trajectory_setpoints is None or self.race_time_setpoints is None:
                if not self._build_race_trajectory(sensor_data):
                    print("can't build race trajectory")
                    return [x, y, self.hover_z, yaw]

            self.race_elapsed += dt
            idx = np.searchsorted(self.race_time_setpoints, self.race_elapsed, side="right") - 1
            idx = int(np.clip(idx, 0, len(self.race_trajectory_setpoints) - 1))
            target = self.race_trajectory_setpoints[idx]
            target_yaw = self._wrap_angle(np.arctan2(target[1] - y, target[0] - x))

            return [float(target[0]), float(target[1]), float(target[2]), target_yaw]

        return [x, y, self.hover_z, yaw]

######
      

    def compute_command(self, sensor_data, camera_data, dt):
        #data shape is a numpy array size (300,300,4) 

        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        #DISCLAIMER : partially reused the code my group did for the Mobile Robots project 
        #(can be found at https://github.com/julienthevenoz/MICRO452_Mobile_Robots/Vision.py)
        corners = self.vision(camera_data, sensor_data)
        target_x, target_y, target_z, target_yaw = self.state_machine(sensor_data, corners, dt)

        return [target_x, target_y, target_z, target_yaw] # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians


# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)

def get_latest_feed():
    return _controller.get_latest_feed()





if __name__ == "__main__":
    pass