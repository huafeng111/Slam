import pr2_utils
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
import math
import random

class Particle():
    def __init__(self,x=0.0,y=0.0,z = 0.0, roll= 0.0, pitch = 0.0, theta=0.0,w=1.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self.yaw = theta
        self.roll = roll
        self.pitch = pitch

class pf_slam():
    def __init__(self):
        # self.time = 0
        self.num_particles = 10
        self.particle_cloud = dict()
        for i in range(self.num_particles):
            self.particle_cloud[i] = Particle()
        self.map = self.init_map()
        _, self.lidar_data = pr2_utils.read_data_from_csv('ECE276A_PR2/code/data/sensor_data/lidar.csv')
        _, self.fog_data = pr2_utils.read_data_from_csv('ECE276A_PR2/code/data/sensor_data/fog.csv')
        _, self.encoder_data = pr2_utils.read_data_from_csv('ECE276A_PR2/code/data/sensor_data/encoder.csv')
        self.Q = np.eye(3)
        self.R = np.eye(3)
        self.S2V_R = np.array([[-0.00680499,-0.0153215,0.99985], [-0.999977,0.000334627,-0.00680066],[-0.000230383,-0.999883,-0.0153234]]).T
        self.V2S_t = np.array([[1.64239,0.247401,1.58411]])
        self.L2V_R = np.array([[0.00130201,0.796097,0.605167],[0.999999 ,-0.000419027 ,-0.00160026],[-0.00102038 ,0.605169 ,-0.796097]]).T
        self.L2V_t = -np.array([[0.8349,-0.0126869,1.76416]])
        self.L2V_t = -np.array([[0.8349,-0.0126869,0]])
        self.V2F_R = np.eye(3)
        self.V2F_t = np.array([[-0.335,-0.035,0.78]])
        self.robot_pose = np.array([[0],[0],[0]])
        self.R2M_R = np.eye(3)
        self.wheel_D_L = 0.623479
        self.wheel_D_R = 0.622806
        self.wheel_base = 1.52439
        self.log_detection_score = np.log(4)
        self.l_max = 1
        self.l_min = -1
        self.max_time = 1000
        self.plot_map_sign = 0


    def pf_main(self,time):
        if time == 0:
            print("time",time)
            self.init_particles([0,0,0])
            self.particle_motion_prediction(time)
            self.particle_weight_update(time)
            self.normalization()
            self.pred_robot_pose()
            self.resample()

        else:
            print("time",time)
            self.particle_motion_prediction(time)
            self.particle_weight_update(time)
            self.normalization()
            self.pred_robot_pose()
            self.resample()

    def init_map(self):
        MAP = {}
        MAP['res']   = 0.1 #meters
        MAP['xmin']  = -50  #meters
        MAP['ymin']  = -50
        MAP['xmax']  =  50
        MAP['ymax']  =  50
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        MAP['map'] = -np.ones((MAP['sizex'],MAP['sizey']),dtype=np.float16) #DATA TYPE: char or int8
        return MAP

    def plot_map(self,MAP):
        fig2 = plt.figure()
        plt.imshow(MAP['map'],cmap="hot");
        plt.title("Occupancy grid map")

    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    def remove_duplicate_row(self,data):
        # data is np.array()
        new_array = [tuple(row) for row in data]
        uniques = np.unique(new_array, axis=0)
        return uniques

    def process_lidar(self, lidar_range,particle):
        particle_x = particle.x
        particle_y = particle.y
        particle_z = particle.z
        particle_roll = particle.roll
        particle_pitch = particle.pitch
        particle_yaw = particle.yaw
        rotation_matrix_roll = np.array([[1,0,0],[0,np.cos(particle_roll), -np.sin(particle_roll)],[0,np.sin(particle_roll), np.cos(particle_roll)]])
        rotation_matrix_yaw = np.array([[np.cos(particle_yaw),-np.sin(particle_yaw), 0],[np.sin(particle_yaw),np.cos(particle_yaw), 0],[0,0,1]])
        rotation_matrix_pitch = np.array([[np.cos(particle_pitch),0,np.sin(particle_pitch)],[0,1,0],[-np.sin(particle_pitch),0,np.cos(particle_pitch)]])
        rotation_matrix_pitch = np.eye(3)
        rotation_matrix_roll = np.eye(3)
        rotation_matrix = self.threemultiple(rotation_matrix_yaw,rotation_matrix_pitch,rotation_matrix_roll)
        # print("particle_x",particle_x,"particle_y",particle_y)
        polar2cart = np.zeros([len(lidar_range),3])
        angles = np.linspace(-5, 185, 286) / 180 * np.pi
        # robot_pose_grid_x = self.robot_pose[0][0]
        # robot_pose_grid_y = self.robot_pose[1][0]

        local_map = self.init_map()
        particle_pose_grid_x = np.ceil((particle_x - local_map['xmin']) / local_map['res'] ).astype(np.int16)-1
        particle_pose_grid_y = np.ceil((particle_y - local_map['ymin']) / local_map['res'] ).astype(np.int16)-1
        temp_obstacle_xy = []
        laser_end_points = np.array([[-1],[-1],[-1]])
        # print("rotation_matrix.T",rotation_matrix.T,"self.L2V_t.T",self.L2V_t.T,"particle_roll",particle_roll)
        for i in range(0, len(lidar_range)):
            if np.logical_and((lidar_range[i] < 80),(lidar_range[i] > 0.1)):
                x = lidar_range[i] * np.cos(angles[i])
                y = lidar_range[i] * np.sin(angles[i])
                temp_xyz = np.array([[x],[y],[0]])
                # print("temp_xyz_0",temp_xyz)
                temp_xyz = self.twomultiple(self.L2V_R,np.array([[x],[y],[0]])) + self.L2V_t.T
                temp_xyz = self.twomultiple(rotation_matrix.T,temp_xyz) + np.array([[particle_x],[particle_y],[0]])
                # print("temp_xyz_1",temp_xyz)
                temp_xyz[2]=0
                # map_xyz = self.twomultiple(self.R2M_R,temp_xyz) - [[self.robot_pose[0][0]],[self.robot_pose[1][0]],[0]]
                map_xyz = temp_xyz
                laser_end_points = np.append(laser_end_points,temp_xyz,axis=1)
                xis = np.ceil((map_xyz[0][0] - local_map['xmin']) / local_map['res'] ).astype(np.int16)-1
                yis = np.ceil((map_xyz[1][0] - local_map['ymin']) / local_map['res'] ).astype(np.int16)-1
                polar2cart[i][0] = xis
                polar2cart[i][1] = yis
                if np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < local_map['sizex'])), (yis < local_map['sizey'])):
                    polar2cart[i][2] = 1 # flag of occupancy
                    temp_obstacle_xy.append([xis,yis])
        empty_grid_x = np.array([])
        empty_grid_y = np.array([])
        for j in range(0,len(temp_obstacle_xy)):
            xj, yj = pr2_utils.bresenham2D(temp_obstacle_xy[j][0],temp_obstacle_xy[j][1],particle_pose_grid_x,particle_pose_grid_y)
            # xj, yj = pr2_utils.bresenham2D(temp_obstacle_xy[j][1],temp_obstacle_xy[j][0],robot_pose_grid_x,robot_pose_grid_y)
            empty_grid_x = np.append(empty_grid_x,xj)
            empty_grid_y = np.append(empty_grid_y,yj)
        empty_grid_x = np.reshape(empty_grid_x,(len(empty_grid_x),1))
        empty_grid_y = np.reshape(empty_grid_y,(len(empty_grid_y),1))
        empty_grid_xy = np.append(empty_grid_x,empty_grid_y,axis=1)
        empty_grid_xy = self.remove_duplicate_row(empty_grid_xy)
        temp_obstacle_xy = self.remove_duplicate_row(np.array(temp_obstacle_xy))
        for grid_pose in empty_grid_xy:
            mx = np.int(grid_pose[0])
            my = np.int(grid_pose[1])
            local_map['map'][mx][my] -= self.log_detection_score
            if local_map['map'][mx][my] >= self.l_max:
                local_map['map'][mx][my] = self.l_max
            elif local_map['map'][mx][my] <= self.l_min:
                local_map['map'][mx][my] = self.l_min

        for grid_pose in temp_obstacle_xy:
            mx = np.int(grid_pose[0])
            my = np.int(grid_pose[1])
            local_map['map'][mx][my] += 2 * self.log_detection_score
            if local_map['map'][mx][my] >= self.l_max:
                local_map['map'][mx][my] = self.l_max
            elif local_map['map'][mx][my] <= self.l_min:
                local_map['map'][mx][my] = self.l_min
        return local_map,laser_end_points

    def particle_motion_prediction(self,time):
        if time == 0:
            avg_forward = 0
        else:
            forward_dist_L = np.pi*(self.wheel_D_L * (self.encoder_data[time,0]-self.encoder_data[time-1,0]))/4800
            forward_dist_R = np.pi*(self.wheel_D_R * (self.encoder_data[time,1]-self.encoder_data[time-1,1]))/4800
            avg_forward = (forward_dist_L + forward_dist_R) / 2

        dtheta = self.fog_data[10*time,2]
        empty_particle_list = []
        for particle_num in self.particle_cloud:
            self.particle_cloud[particle_num].x += avg_forward * np.cos(self.particle_cloud[particle_num].yaw) + avg_forward*(2*random.random()-1)
            self.particle_cloud[particle_num].y += avg_forward * np.sin(self.particle_cloud[particle_num].yaw) + avg_forward*(2*random.random()-1)
            self.particle_cloud[particle_num].yaw += dtheta
        # for i in range(self.num_particles):
        #     print("self.particle_cloud[i]",self.particle_cloud[i].w)

    def init_particles(self, xy_theta=None):
        rad = 1 # meters
        self.particle_cloud = dict()
        self.particle_cloud[len(self.particle_cloud)] = (Particle(xy_theta[0], xy_theta[1],0,0,0, xy_theta[2],1/self.num_particles))
        # print("self.particle_cloud[0]",self.particle_cloud[0],self.particle_cloud[0].x)
        for i in range(self.num_particles-1):
            theta = random.random() * 2 * np.pi
            theta_xy = random.random() * 2 * np.pi
            radius = random.random() * rad
            x = radius * np.sin(theta_xy) + xy_theta[0]
            y = radius * np.cos(theta_xy) + xy_theta[1]
            particle = Particle(x,y,0,0,0,theta,1/self.num_particles)
            self.particle_cloud[i+1] = particle


    def pred_robot_pose(self):
        x = 0
        y = 0
        theta = 0
        angles = [0,0]
        for particle_num in self.particle_cloud:
            x += self.particle_cloud[particle_num].x * self.particle_cloud[particle_num].w
            y += self.particle_cloud[particle_num].y * self.particle_cloud[particle_num].w
            v = [self.particle_cloud[particle_num].w * np.cos(self.particle_cloud[particle_num].yaw), self.particle_cloud[particle_num].w * np.sin(self.particle_cloud[particle_num].yaw)]
            angles = [angles[0]+ v[0], angles[1]+ v[1]]
        x_sign = angles[0]>=0
        y_sign = angles[1]>=0
        if np.abs(angles[0])<=0.001:
            if y_sign>=0:
                theta = math.radians(90)
            if y_sign<0:
                theta = math.radians(-90)
        else:
            theta = np.arctan(angles[1]/angles[0])
        if x_sign<=0:
            theta = theta
        else:
            theta = theta + np.pi
        self.robot_pose = [x, y, theta]
        if self.plot_map_sign:
            particle = Particle(x, y,0,0,0,theta)
            temp_map,laser_end_points = self.process_lidar(self.lidar_data[self.time,:],particle)
            self.plot_map(temp_map)
            time.sleep(5)

    # def sum_vectors(self,angles):

    def particle_weight_update(self,time):
        tot_weight = 0
        x_list = []
        y_list = []
        yaw_list = []
        w_list = []
        for particle_num in range(len(self.particle_cloud)):
            particle = self.particle_cloud[particle_num]
            temp_map,laser_end_points = self.process_lidar(self.lidar_data[time,:],particle)

            # print("max",np.max(temp_map['map']))
            xis = np.ceil((particle.x - temp_map['xmin']) / temp_map['res'] ).astype(np.int16)-1
            yis = np.ceil((particle.y - temp_map['ymin']) / temp_map['res'] ).astype(np.int16)-1
            indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < temp_map['sizex'])), (yis < temp_map['sizey']))
            temp_map['map'][xis[indGood],yis[indGood]] = 1

            x_im = np.arange(temp_map['xmin'],temp_map['xmax']+temp_map['res'],temp_map['res']) #x-positions of each pixel of the map
            y_im = np.arange(temp_map['ymin'],temp_map['ymax']+temp_map['res'],temp_map['res']) #y-positions of each pixel of the map
            Y=laser_end_points[0:2,1:laser_end_points.shape[0]]
            x_range = np.arange(-0.4,0.4+0.1,0.1)
            y_range = np.arange(-0.4,0.4+0.1,0.1)
            weight = np.abs(pr2_utils.mapCorrelation(temp_map['map'],x_im,y_im,Y,x_range,y_range))
            min_index_x,min_index_y = np.where(weight == np.min(weight))
            min_index_x = min_index_x - 5
            min_index_y = min_index_y - 5
            min_index_xy = min_index_x + min_index_y
            temp_min = np.where(min_index_xy == np.min(min_index_xy))
            dx = min_index_x[temp_min[0][0]]
            dy = min_index_y[temp_min[0][0]]
            tot_weight += np.max(weight)
            w_list.append(np.max(weight))
            x_list.append(particle.x + dx * 0.1)
            y_list.append(particle.y + dy * 0.1)

        for i in range(len(self.particle_cloud)):
            self.particle_cloud[i].x = x_list[i]
            self.particle_cloud[i].y = y_list[i]
            self.particle_cloud[i].w = w_list[i]



    def normalization(self):
        """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0) """
        print("count")
        tot_weight = 0
        for particle_num in self.particle_cloud:
            tot_weight = tot_weight + self.particle_cloud[particle_num].w

        weight_list = []
        for i in range(len(self.particle_cloud)):
            weight_list.append(self.particle_cloud[i].w/tot_weight)
        for particle_num in range(len(self.particle_cloud)):
            # print("particle_num0",particle_num)
            self.particle_cloud[particle_num].w = weight_list[particle_num]
            print("self.particle_cloud[particle_num].w",self.particle_cloud[particle_num].w)


    def resample(self):
        newParticles = dict()
        for i in range(len(self.particle_cloud)):
            choice = 1/self.num_particles
            csum = 0 # cumulative sum
            for particle_num in self.particle_cloud:
                csum += self.particle_cloud[particle_num].w
                if csum >= choice:
                    newParticles[len(newParticles)] = self.particle_cloud[particle_num]
                    break
        len_newParticles = len(newParticles)
        for i in range(self.num_particles - len(newParticles)):
            theta = random.random() * 2 * np.pi
            theta_xy = random.random() * 2 * np.pi
            radius = random.random() * 1
            x = radius * np.sin(theta_xy) + self.robot_pose[0]
            y = radius * np.cos(theta_xy) + self.robot_pose[1]
            one_particle = Particle(x,y,0,0,0,theta,0)
            newParticles[len_newParticles+i] = one_particle
        self.particle_cloud = newParticles

    def twomultiple(self,a,b):
        result = np.matmul(a , b)
        return result

    def threemultiple(self,a,b,c):
        result = np.matmul(np.matmul(a , b), c)
        return result

if __name__ == '__main__':

    # pr2_utils.show_lidar()
    pf_slam = pf_slam()
    # pf_slam.process_lidar(pf_slam.lidar_data[0, :])
    # fig1 = plt.figure()
    # plt.imshow(pf_slam.map['map'],cmap="hot")
    # plt.title("Occupancy grid map1")
    # pf_slam.process_lidar(pf_slam.lidar_data[0, :])
    for time in range(10):
        pf_slam.pf_main(time)
        time +=1




