"""
CBF Controller for Isaac Gym
Adapted from MuJoCo CBF implementation
"""

import casadi as ca
import numpy as np
import torch
from termcolor import colored
import time
import sys
import os

class DISTURBANCE_OBSERVER:
    def __init__(self,T_step,alpha,f,g1,g2,w0):
        self.u_len=len(f)
        self.T_step = T_step
        self.f=f
        self.g1=g1
        self.g2=g2
        self.dt=np.zeros(self.u_len).reshape(self.u_len, 1)

        self.a1=1.1
        self.a2=1.5
        self.c = 10*np.array([10,10,10,10,10,10,10,10])
        self.a = self.a1*self.c
        self.l = self.a2*np.sqrt(self.c)
        self.z1 = np.zeros(self.u_len).reshape(self.u_len, 1)
        self.z2 = np.zeros(self.u_len).reshape(self.u_len, 1)
        self.x2 = np.zeros(self.u_len).reshape(self.u_len, 1)
        self.x1 = np.zeros(self.u_len).reshape(self.u_len, 1)
    # def update_z(self,ut):
    #     self.zt = self.zt-self.alpha*self.ld @ (self.f+self.g1@ut+self.g2@self.dt)*self.T_step
    
    def update_d(self,ut,currrent_state):
        try:
            for i in range(self.u_len):
                error=currrent_state[i]-self.x1[i]
                self.dt[i] = error
                self.z1[i] = self.l[i]*np.sqrt(abs(error))*np.sign(error)
                self.z2[i] = self.a[i]*np.sign(error)
                self.x1[i] = self.x1[i] + self.T_step*(0+ut[i]+self.x2[i]+self.z1[i])
                self.x2[i] = self.x2[i] + self.T_step*self.z2[i]
        except Exception as e:
            print(e)
        return self.dt.reshape(self.u_len, 1)

class RECTANGLE:
    def __init__(self,C,V0,V1,V2):
        self.C = C
        self.V0 = V0
        self.V1 = V1
        self.V2 = V2
        self.E0 = V1-V0
        self.E1 = V0-V2
        self.V3 = V2+self.E0
        self.e0 = ca.norm_2(self.E0)/2
        self.e1 = ca.norm_2(self.E1)/2
        self.u0 = self.E0/(2*self.e0)
        self.u1 = self.E1/(2*self.e1)
        self.u2 = ca.cross(self.u0,self.u1) 
# 添加SuppressOutput类来禁用qpOASES的打印输出
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class CBF_controller:
    def __init__(self,obstacle_type_num=[0,1,3],
    T_step=0.01,O_T_step=0.01,h_threshold=0.08,CBF_mode='11',out_limite=None
    ,use_statistic_obstacle=False):
        ###obstacle
        start_time = time.time()
        self.CBF_mode = CBF_mode
        self.T_step = T_step
        self.O_T_step = O_T_step
        print(f'self.CBF_mode = {self.CBF_mode}')
        self.u_len=8
        self.n_conpoment=7
        self.n_base_cbf=3
        if use_statistic_obstacle ==False:
            self.use_statistic_obstacle=False
            self.statistic_capsule_obstacle_num=0
            self.n_env_limite_cbf=4
        else: 
            self.use_statistic_obstacle=True
            self.statistic_capsule_obstacle_num=4
            self.n_env_limite_cbf=8  #高于地面及在范围内
        
        sphere_obstacle_num = obstacle_type_num[0]
        capsule_obstacle_num = obstacle_type_num[1]
        rectangle_obstacle_num = obstacle_type_num[2]
        self.h_threshold = h_threshold 
        self.h_danger = 0.0018  #<h_threshold 0.018
        self.normal_gama =20  #1.4 
        self.h_danger_p_gama = -100  #<0 -2
        self.h_danger_n_gama = 500 #>0 100
        if CBF_mode=='00':
            self.h_danger_p_gama = -2
            self.h_danger_n_gama = 5 #>0 2 400 600 10000
        if CBF_mode=='10':
            self.h_danger_p_gama = -20
            self.h_danger_n_gama = 20 #>0 2 400 600 10000
        if CBF_mode=='01':
            self.h_danger_p_gama = -20
            self.h_danger_n_gama = 200 #>0 2 400 600 10000
        if CBF_mode=='11':
            self.h_danger_p_gama = -100
            self.h_danger_n_gama = 600 #>0 2 200 400 600 10000  600best 800
        
        self.sphere_obstacle_num=sphere_obstacle_num
        self.sphere_obstacle_list=ca.SX.sym('sphere_obstacle_list',sphere_obstacle_num,3) #position of sphere obs
        self.capsule_obstacle_num=capsule_obstacle_num
        self.capsule_obstacle_list=ca.SX.sym('capsule_obstacle_list',capsule_obstacle_num,6) #position of capsule obs
        self.rectangle_obstacle_num=rectangle_obstacle_num
        self.rectangle_obstacle_list=ca.SX.sym('rectangle_obstacle_list',rectangle_obstacle_num,12) #position of rectangle obs:center\v0\v1\v2

        self.base_height=0.0
        # self.statistic_capsule_obstacle_list=ca.SX.sym('statistic_capsule_obstacle_list',self.statistic_capsule_obstacle_num,6) #position of statistic capsule obs
        self.statistic_capsule_obstacle_list=ca.SX([
                                                    [0,3.1,0.90824-self.base_height,-1.6,3.1,0.091763-self.base_height],
                                                    [0,3.1,0.90824-self.base_height,1.6,3.1,0.091763-self.base_height],
                                                    [-1.25,5.5,0.7-self.base_height,-1.25,4.5,0.7-self.base_height],
                                                    [1.25,5.5,0.7-self.base_height,1.25,4.5,0.7-self.base_height],
                                                    ]) #position of statistic capsule obs
        self.r_arm = np.array([.036,.029,.029,.029,.029,.029,0.4])
        self.r_safe_expand = 0.01 #0.04
        self.capsule_r1,self.capsule_r2,self.capsule_r3,self.capsule_r4=0.05,0.05,0.55,0.55
        self.safe_statistic_R_list = np.array( [self.r_arm+self.capsule_r1+2*self.r_safe_expand,
                                                self.r_arm+self.capsule_r2+2*self.r_safe_expand,
                                                self.r_arm+self.capsule_r3+2*self.r_safe_expand,
                                                self.r_arm+self.capsule_r4+2*self.r_safe_expand]) 

        self.n_statistic_cbf=self.n_env_limite_cbf+self.statistic_capsule_obstacle_num*self.n_conpoment


        self.base_posture = ca.SX.sym('base_posture',3)      #base_posture
        self.piper_angles = ca.SX.sym('piper_angles',5)                #angles of joints
        self.state=ca.vertcat(self.base_posture,self.piper_angles)
        # self.angles = ca.SX.sym('angles',self.u_len)                #angles of joints 
        # self.state=self.angles              #为了匹配后面函数中的写法
        self.state_velocity = ca.SX.sym('state_velocity',self.u_len)   
        self.safe_R_list=ca.SX.sym('safe_R_list',sphere_obstacle_num+capsule_obstacle_num+rectangle_obstacle_num,self.n_conpoment)       #safe radius of obs
        self.R_safe_base=ca.SX([0.3])
        self.totle_num=self.sphere_obstacle_num+self.capsule_obstacle_num+self.rectangle_obstacle_num
        self.obs_v_list = ca.SX.sym('obs_v_list',self.totle_num,3)
        self.n_CBF_constrain=self.totle_num*self.n_conpoment + self.n_base_cbf+self.n_statistic_cbf
        ###system
        self.T_step = T_step
        self.u  = ca.SX.sym('u',self.u_len,1)
        self.Ax = ca.SX.zeros(self.u_len,1)
        self.gx = ca.SX.eye(self.u_len)
        self.g2 = ca.SX.eye(self.u_len)

        
        self.ZERO=0.000001
        self.gama = ca.SX.sym('gama',1)
        self.alpha = 20
        self.w0=0.6
        self.w1=0.4
        self.dt = np.zeros(self.u_len).reshape(self.u_len, 1)
        self.v1=12
        self.turning_beta = 10
        self.param1 = np.zeros(self.n_CBF_constrain)
        self.param2 = np.zeros(self.n_CBF_constrain)
        self.sigma_v = 0.18 #0.18 0.23
        self.h=ca.SX.zeros(self.n_CBF_constrain, 1)
        self.dhdx=ca.SX.zeros(self.n_CBF_constrain, self.u_len)
        self.dhdp=ca.SX(self.n_CBF_constrain, 3)
        self.H=ca.SX.zeros(self.n_CBF_constrain, 1)
        self.G=ca.SX.zeros(self.n_CBF_constrain, self.u_len)
        self.K=ca.SX.zeros(self.n_CBF_constrain, self.u_len)
        self.dis2obs=ca.SX.zeros(self.n_CBF_constrain, 1)
        
        self.F_H_list = []
        self.F_G_list = []
        self.F_K_list = []
        self.u_last=None

        self.cal_H=ca.DM.zeros((self.n_CBF_constrain, 1))
        self.cal_G=ca.DM.zeros((self.n_CBF_constrain, self.u_len))
        self.cal_K=ca.DM.zeros((self.n_CBF_constrain, self.u_len))

        if out_limite is None:
            self.output_limite=np.array([0.5, 0.5, 1.0, 3.14, 3.40, 3.14, 3.93, 3.93])
        else:
            self.output_limite=np.array(out_limite)


        self.H_mat_para = ca.DM([
                            200, 200, 200,   # base 线速度锁死
                            0.5, 0.2, 0.3, 0.3, 0.3  # 肩<肘<腕
                        ])  #[200,200,10,2,2,0.8,0.3,0.3] [20,20,10,1,1,1,1,0.3]
        self.H_mat =  ca.diag(self.H_mat_para)
        self.g_list=ca.SX.zeros(self.n_CBF_constrain+4*self.u_len+self.n_CBF_constrain, 1) # self.n_CBF_constrain + output limite + state limite + slack
        self.lbg=[0]*(self.n_CBF_constrain+4*self.u_len+self.n_CBF_constrain)
        self.slack = ca.SX.sym('slack',self.n_CBF_constrain)
        self.g_list[(self.n_CBF_constrain+4*self.u_len):] = self.slack
        self.u_augmented = ca.vertcat(self.u, self.slack)
        self.H_mat_augmented = ca.diag(ca.horzcat(self.H_mat_para.T,
                                       100000*ca.DM.ones(self.slack.size1()).T))
        self.is_slack=True
        self.low_joint_limite = np.array([-10,-10,-5,
                                          -2.68,0,-2.697,-1.832,-1.22])
        self.up_joint_limite  = np.array([10,10,5,
                                          2.68,3.14,0,1.832,1.22])  #UR5+gripper/ur5_gripper.urdf

        # self.low_joint_limite = np.array([-10,-10,-5,
        #                                   -3.14,-3.14,-3.14,-3.14,-3.14])
        # self.up_joint_limite  = np.array([10,10,5,
        #                                   3.14,3.14,3.14,3.14,3.14])  #UR5+gripper/ur5_gripper.urdf

        self.get_solution=False
        self.initial_MobileARM()
        self.initial_CBF()
        self._initialize_functions()
        print(f"finish CBF initialize,used time = {time.time()-start_time}")

    def initial_MobileARM(self):

        cos_y = ca.cos(self.base_posture[2]+self.state_velocity[2]*self.T_step)
        sin_y = ca.sin(self.base_posture[2]+self.state_velocity[2]*self.T_step)

        cos_0,cos_1,cos_2,cos_3,cos_4 = ca.cos(self.piper_angles[0]+self.state_velocity[3]*self.T_step),ca.cos(self.piper_angles[1]+self.state_velocity[4]*self.T_step),ca.cos(self.piper_angles[2]+self.state_velocity[5]*self.T_step),ca.cos(self.piper_angles[3]+self.state_velocity[6]*self.T_step),ca.cos(self.piper_angles[4]+self.state_velocity[7]*self.T_step)
        sin_0,sin_1,sin_2,sin_3,sin_4 = ca.sin(self.piper_angles[0]+self.state_velocity[3]*self.T_step),ca.sin(self.piper_angles[1]+self.state_velocity[4]*self.T_step),ca.sin(self.piper_angles[2]+self.state_velocity[5]*self.T_step),ca.sin(self.piper_angles[3]+self.state_velocity[6]*self.T_step),ca.sin(self.piper_angles[4]+self.state_velocity[7]*self.T_step)
         
        ############   ROBOT CAR JOINT MATRIX

        #####TRANSITION XY OF CAR
        self.T_XY_CAR=ca.SX.eye(4)
        T_XY_CAR_list =[[1, 0, 0, self.base_posture[0]+self.state_velocity[0]*self.T_step],
                        [0, 1, 0, self.base_posture[1]+self.state_velocity[1]*self.T_step],
                        [0, 0, 1, 0.26],
                        [0, 0, 0, 1]] 
        for i in range(16):
            self.T_XY_CAR[i] = T_XY_CAR_list[i%4][i//4]

        #####ROTATION Z OF CAR
        T_yaw = ca.SX.eye(4)
        T_yaw_list = [[cos_y, -sin_y, 0., 0.],
                        [sin_y, cos_y, 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]]
        for i in range(16):
            T_yaw[i] = T_yaw_list[i%4][i//4]

        # 组合变换（注意顺序：roll -> pitch -> yaw）
        T_rotation = T_yaw 

        self.T_world_base = self.T_XY_CAR @ T_rotation
        ############   ROBOT ARM JOINT MATRIX
        ############   机械臂部分


        ##### 第一段 base-> J1
        T_base_A0 = ca.SX([[1.,0.,0.,0.],
                           [0.,1.,0.,0.],
                           [0.,0.,1.,0.074],
                           [0.,0.,0.,1.]])
        T_A0_A = ca.SX.eye(4)
        T_A0_A_list = [[cos_0, -sin_0, 0., 0.],
                        [sin_0, cos_0, 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]]
        for i in range(16):
            T_A0_A[i] = T_A0_A_list[i%4][i//4]
        self.T_base_A =T_base_A0 @ T_A0_A
        #print(T_base_A0)
        #print(T_A0_A)
        #print('T_base_A = ',T_base_A)
        ##### 第二段 J1->J2
        T_A_B0 = ca.SX([[1.0, 0.0, 0.0, 0.00],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.049],
                        [0. ,0. ,0. ,1.]])
        T_B0_B = ca.SX.eye(4)
        T_B0_B_list = [[cos_1, 0, sin_1, 0.],
                        [0, 1, 0., 0.],
                        [-sin_1, 0., cos_1, 0.],
                        [0., 0., 0., 1.]]
        for i in range(16):
            T_B0_B[i] = T_B0_B_list[i%4][i//4]
        self.T_A_B =T_A_B0 @ T_B0_B
        #print('T_A_B = ',T_A_B)
        ##### 第三段 J2->J3
        T_B_C0 = ca.SX([[1,0,0,-0.28],
                        [0,1,0,0.0],
                        [0,0,1,0.045],
                        [0,0,0,1]])
        T_C0_C = ca.SX.eye(4)
        T_C0_C_list = [[cos_2, 0, sin_2, 0.],
                        [0, 1, 0., 0.],
                        [-sin_2, 0., cos_2, 0.],
                        [0., 0., 0., 1.]]
        for i in range(16):
            T_C0_C[i] = T_C0_C_list[i%4][i//4]
        self.T_B_C =T_B_C0 @ T_C0_C
        #print('T_B_C = ',T_B_C)
        ##### 第四段 J3->J4
        T_C_D0 = ca.SX([[1,0,0,0.22],
                        [0,1,0,0],
                        [0,0,1,0.025],
                        [0,0,0,1]])
        T_D0_D = ca.SX.eye(4)
        T_D0_D_list = [[1.,0.,0.,0.],  # X轴不变
                        [0.,cos_3,-sin_3,0.],  # Y和Z绕X轴旋转
                        [0.,sin_3,cos_3,0.],
                        [0.,0.,0.,1.]]
        for i in range(16):
            T_D0_D[i] = T_D0_D_list[i%4][i//4]
        self.T_C_D =T_C_D0 @ T_D0_D
        #print('T_C_D = ',T_C_D)
        ##### 第五段 J4->J5
        T_D_E0 = ca.SX([[1,0,0,0.036],
                        [0,1,0,0.0],
                        [0,0,1,0],
                        [0,0,0,1]])
        T_E0_E = ca.SX.eye(4)
        T_E0_E_list = [[cos_4, 0, sin_4, 0.],
                        [0, 1, 0., 0.],
                        [-sin_4, 0., cos_4, 0.],
                        [0., 0., 0., 1.]]
        for i in range(16):
            T_E0_E[i] = T_E0_E_list[i%4][i//4]
        self.T_D_E =T_D_E0 @ T_E0_E
        #print('T_D_E = ',T_D_E)
        ##### 第六段 J5->end_effoctor
        T_E_F0 = ca.SX([[1,0,0,0.236],
                        [0,1,0,0.0],
                        [0,0,1,0.0],
                        [0,0,0,1]])
        self.T_E_END = T_E_F0

        # world_P_BASE = self.T_XY_CAR @ self.T_car_BASE
        # world_P_A = world_P_BASE @ self.T_base_A
        # world_P_B = world_P_BASE @ self.T_base_A @ self.T_A_B
        # world_P_C = world_P_BASE @ self.T_base_A @ self.T_A_B @ self.T_B_C
        # world_P_D = world_P_BASE @ self.T_base_A @ self.T_A_B @ self.T_B_C @ self.T_C_D
        # world_P_E = world_P_BASE @ self.T_base_A @ self.T_A_B @ self.T_B_C @ self.T_C_D @ self.T_D_E
        # world_P_END = world_P_BASE @ self.T_base_A @ self.T_A_B @ self.T_B_C @ self.T_C_D @ self.T_D_E @ self.T_E_END

        BASE_T_BASE = ca.SX([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
        BASE_T_A = self.T_world_base @ self.T_base_A @ BASE_T_BASE
        BASE_T_B = self.T_world_base @ self.T_base_A @ self.T_A_B @ BASE_T_BASE
        BASE_T_C = self.T_world_base @ self.T_base_A @ self.T_A_B @ self.T_B_C
        BASE_T_D = self.T_world_base @ self.T_base_A @ self.T_A_B @ self.T_B_C @ self.T_C_D
        BASE_T_E = self.T_world_base @ self.T_base_A @ self.T_A_B @ self.T_B_C @ self.T_C_D @ self.T_D_E
        BASE_T_END = self.T_world_base @ self.T_base_A @ self.T_A_B @ self.T_B_C @ self.T_C_D @ self.T_D_E @ self.T_E_END


        self.A = BASE_T_A[0:3,3]
        self.B = BASE_T_B[0:3,3]
        self.C = BASE_T_C[0:3,3]
        self.D = BASE_T_D[0:3,3]
        self.E = BASE_T_E[0:3,3]
        self.END = BASE_T_END[0:3,3]

        
        self.arm_point = ca.horzcat(self.A,self.B,self.C,self.D,self.E,self.END)
        ##############  CALCULATE DISTANCE BETWEEN POINT AND LINE SEGMENT   
        # WE HAVE: Obstacle_list,base_P_A,base_P_B,base_P_C0,base_P_C1,base_P_D,base_P_E,base_P_F,base_P_END

        #BASE
        self.BASE = ca.SX(3,1)
        self.BASE[0] = self.base_posture[0]+self.state_velocity[0]*self.T_step
        self.BASE[1] = self.base_posture[1]+self.state_velocity[1]*self.T_step
        self.BASE[2] = 0.26

        #BASEA
        self.BASEA = self.A - self.BASE
        self.BASEA_norm = ca.norm_2(self.BASEA)
        #AB
        self.AB = self.B - self.A
        self.AB_norm = ca.norm_2(self.AB)
        #BC
        self.BC = self.C - self.B
        self.BC_norm = ca.norm_2(self.BC)
        #CD
        self.CD = self.D - self.C
        self.CD_norm = ca.norm_2(self.CD)
        #DE
        self.DE = self.E - self.D
        self.DE_norm = ca.norm_2(self.DE)
        #EEND
        self.EEND = self.END - self.E
        self.EEND_norm = ca.norm_2(self.EEND)


    def _initialize_functions(self):
        # self.F_arm_point = ca.Function('F_armpoint',[self.state],[self.arm_point])
        self.F_dis2obs = ca.Function('dis2obs',
            [self.state,self.sphere_obstacle_list,self.capsule_obstacle_list,self.rectangle_obstacle_list,self.obs_v_list,self.state_velocity],
            [self.dis2obs])                   
        self.F_barriers = ca.Function('F_barriers',
            [self.state,self.safe_R_list,self.sphere_obstacle_list,self.capsule_obstacle_list,self.rectangle_obstacle_list,self.obs_v_list,self.state_velocity],
            [self.h])
        self.F_dhdx = ca.Function('F_B_D',
            [self.state,self.safe_R_list,self.sphere_obstacle_list,self.capsule_obstacle_list,self.rectangle_obstacle_list,self.obs_v_list,self.state_velocity]
            ,[self.dhdx])
        self.F_H = ca.Function('F_H',
            [self.state,self.safe_R_list,self.sphere_obstacle_list,self.capsule_obstacle_list,self.rectangle_obstacle_list,self.gama,self.obs_v_list,self.state_velocity]
            ,[self.H])
        self.F_G = ca.Function('F_G',
            [self.state,self.safe_R_list,self.sphere_obstacle_list,self.capsule_obstacle_list,self.rectangle_obstacle_list,self.obs_v_list,self.state_velocity]
            ,[self.G])
        self.F_K = ca.Function('F_K',
            [self.state,self.safe_R_list,self.sphere_obstacle_list,self.capsule_obstacle_list,self.rectangle_obstacle_list,self.obs_v_list,self.state_velocity]
            ,[self.K])

        for i in range(self.n_CBF_constrain):
            self.F_H_list.append(ca.Function('H_fun'+f'_{i}',
            [self.state,self.safe_R_list,self.sphere_obstacle_list,self.capsule_obstacle_list,self.rectangle_obstacle_list,self.gama,self.obs_v_list,self.state_velocity]
            ,[self.H[i]]))
            self.F_G_list.append(ca.Function('G_fun'+f'_{i}',
            [self.state,self.safe_R_list,self.sphere_obstacle_list,self.capsule_obstacle_list,self.rectangle_obstacle_list,self.obs_v_list,self.state_velocity]
            ,[self.G[i,:]]))
            self.F_K_list.append(ca.Function('K_fun'+f'_{i}',
            [self.state,self.safe_R_list,self.sphere_obstacle_list,self.capsule_obstacle_list,self.rectangle_obstacle_list,self.obs_v_list,self.state_velocity]
            ,[self.K[i,:]]))
                
    def set_beta(self,beta):
        self.beta   = beta*self.turning_beta
        self.param1 = self.w1**2/(2*self.v1*self.beta) #w1^2/(2v1*beta)
        self.param2 = self.beta/(4*self.alpha-2*self.v1-2*self.gama) #beta/(4alpha-2v1-2gama)

    def distance_point_to_point(self,point1,point2):
        return ca.norm_2(point1-point2)

    def distance_point_to_segment(self,point,line_start,line_end,line_direction,line_direction_norm):
        p0p2 = point- line_end #1 END
        p0p1 = point - line_start #2 START
        t = ca.dot(p0p1,line_direction) / (line_direction_norm**2)#2
        projection = t * line_direction
        distance_vector = p0p1 - projection #2
        distance = ca.norm_2(distance_vector)
        d2segment_1 = ca.if_else(t <= 0, ca.norm_2(p0p1), 0) #2
        d2segment_2 = ca.if_else(t > 0, ca.if_else(t < 1,distance, 0) ,0)
        d2segment_3 = ca.if_else(t >= 1, ca.norm_2(p0p2), 0) #1
        return d2segment_1+d2segment_2+d2segment_3

    def distance_segment_to_segment(self,Q0,Q1,P0,P1,P0P1,P0P1_norm): #0:START 1:END  P0P1 arm , Q0Q1 CAPSULE OBSTACLE
        #P0P1=P1-P0
        P0P1=P0P1
        P0P1_norm=P0P1_norm#ca.norm_2(P0P1)
        Q0Q1=Q1-Q0
        Q0Q1_norm=ca.norm_2(Q0Q1)
        Q0P0=P0-Q0
        P1Q1=Q1-P1
        P1Q1_norm=ca.norm_2(P1Q1)
        QOP1=P1-Q0
        QOP1_norm=ca.norm_2(QOP1)
        Q1P0=P0-Q1
        Q1P0_norm=ca.norm_2(Q1P0)
        Q0P0=P0-Q0
        Q0P0_norm=ca.norm_2(Q0P0)
        a=ca.dot(P0P1,P0P1)
        b=ca.dot(P0P1,Q0Q1)
        c=ca.dot(Q0Q1,Q0Q1)
        d=ca.dot(P0P1,Q0P0)
        e=ca.dot(Q0Q1,Q0P0)
        f=ca.dot(Q0P0,Q0P0)
        parallel_flag = a*c-b*b
        s_bar = ca.if_else(parallel_flag > self.ZERO,(b*e - c*d)/parallel_flag ,self.ZERO)
        t_bar = ca.if_else(parallel_flag > self.ZERO,(a*e - b*d)/parallel_flag ,self.ZERO)
        ###1.1 s>1 t>1
        d2segment_1 = ca.if_else(parallel_flag > self.ZERO,
                      ca.if_else(s_bar>1,
                      ca.if_else(t_bar>1,
                      ca.if_else(-b+c-e>=0,  #Dt
                      ca.if_else(a-b+d<0,    #Ds
                      self.distance_point_to_segment(P1,Q0,Q1,Q0Q1,Q0Q1_norm),0),0),0),0),0)
        d2segment_2 = ca.if_else(parallel_flag > self.ZERO,
                      ca.if_else(s_bar>1,
                      ca.if_else(t_bar>1,
                      ca.if_else(-b+c-e<0,
                      ca.if_else(a-b+d>=0, 
                      self.distance_point_to_segment(Q1,P0,P1,P0P1,P0P1_norm),0),0),0),0),0)
        d2segment_3 = ca.if_else(parallel_flag > self.ZERO,
                      ca.if_else(s_bar>1,
                      ca.if_else(t_bar>1,
                      ca.if_else(-b+c-e<0,
                      ca.if_else(a-b+d<0, 
                      P1Q1_norm,0),0),0),0),0)        
        ###1.2  s>1 0<=t<=1
        d2segment_4 = ca.if_else(parallel_flag > self.ZERO,
                      ca.if_else(s_bar>1,
                      ca.if_else(t_bar>=0,
                      ca.if_else(t_bar<=1,
                      self.distance_point_to_segment(P1,Q0,Q1,Q0Q1,Q0Q1_norm),0),0),0),0)
        ###1.3  s>1 t<0
        d2segment_5 = ca.if_else(parallel_flag > self.ZERO,
                      ca.if_else(s_bar>1,
                      ca.if_else(t_bar<0,
                      ca.if_else(b+e>=0,
                      ca.if_else(a+d<0, 
                      self.distance_point_to_segment(P1,Q0,Q1,Q0Q1,Q0Q1_norm),0),0),0),0),0)
        d2segment_6 = ca.if_else(parallel_flag > self.ZERO,
                      ca.if_else(s_bar>1,
                      ca.if_else(t_bar<0,
                      ca.if_else(b+e<0,
                      ca.if_else(a+d>=0, 
                      self.distance_point_to_segment(Q0,P0,P1,P0P1,P0P1_norm),0),0),0),0),0)
        d2segment_7 = ca.if_else(parallel_flag > self.ZERO,
                      ca.if_else(s_bar>1,
                      ca.if_else(t_bar<0,
                      ca.if_else(b+e<0,
                      ca.if_else(a+d<0, 
                      QOP1_norm,0),0),0),0),0)          
        ###1.4 0<=s<=1 t>1
        d2segment_8 = ca.if_else(parallel_flag > self.ZERO,
                      ca.if_else(s_bar>=0,
                      ca.if_else(s_bar<=1,
                      ca.if_else(t_bar>1,
                      self.distance_point_to_segment(Q1,P0,P1,P0P1,P0P1_norm),0),0),0),0)
        ###1.5 0<=s<=1 0<=t<=1
        d2segment_9 = ca.if_else(parallel_flag > self.ZERO,
                      ca.if_else(s_bar>=0,
                      ca.if_else(s_bar<=1,
                      ca.if_else(t_bar>=0,
                      ca.if_else(t_bar<=1,
                      ca.sqrt(a*s_bar*s_bar-2*b*s_bar*t_bar+c*t_bar*t_bar+2*d*s_bar-2*e*t_bar+f)
                      ,0),0),0),0),0)
        ###1.6 0<=s<=1 t<0
        d2segment_10 = ca.if_else(parallel_flag > self.ZERO,
                       ca.if_else(s_bar>=0,
                       ca.if_else(s_bar<=1,
                       ca.if_else(t_bar<0,
                       self.distance_point_to_segment(Q0,P0,P1,P0P1,P0P1_norm),0),0),0),0)
        ###1.7 s<0 t>1
        d2segment_11 = ca.if_else(parallel_flag > self.ZERO,
                       ca.if_else(s_bar<0,
                       ca.if_else(t_bar>1,
                       ca.if_else(c-e>=0,
                       ca.if_else(b-d<0, 
                       self.distance_point_to_segment(P1,Q0,Q1,Q0Q1,Q0Q1_norm),0),0),0),0),0)
        d2segment_12 = ca.if_else(parallel_flag > self.ZERO,
                       ca.if_else(s_bar<0,
                       ca.if_else(t_bar>1,
                       ca.if_else(c-e<0,
                       ca.if_else(b-d>=0, 
                       self.distance_point_to_segment(Q1,P0,P1,P0P1,P0P1_norm),0),0),0),0),0)
        d2segment_13 = ca.if_else(parallel_flag > self.ZERO,
                       ca.if_else(s_bar<0,
                       ca.if_else(t_bar>1,
                       ca.if_else(c-e<0,
                       ca.if_else(b-d<0,  
                       Q1P0_norm,0),0),0),0),0)
        ###1.8 s<0 0<=t<=1
        d2segment_14 = ca.if_else(parallel_flag > self.ZERO,
                       ca.if_else(s_bar<0,
                       ca.if_else(t_bar>=0,
                       ca.if_else(t_bar<=1,
                       self.distance_point_to_segment(P0,Q0,Q1,Q0Q1,Q0Q1_norm),0),0),0),0)
        ###1.9  s<0 t<0
        d2segment_15 = ca.if_else(parallel_flag > self.ZERO,
                       ca.if_else(s_bar<0,
                       ca.if_else(t_bar<0,
                       ca.if_else(e>=0,
                       ca.if_else(-d<0, 
                       self.distance_point_to_segment(P0,Q0,Q1,Q0Q1,Q0Q1_norm),0),0),0),0),0)
        d2segment_16 = ca.if_else(parallel_flag > self.ZERO,
                       ca.if_else(s_bar<0,
                       ca.if_else(t_bar<0,
                       ca.if_else(e<0,
                       ca.if_else(-d>=0, 
                       self.distance_point_to_segment(Q0,P0,P1,P0P1,P0P1_norm),0),0),0),0),0)
        d2segment_17 = ca.if_else(parallel_flag > self.ZERO,
                       ca.if_else(s_bar<0,
                       ca.if_else(t_bar<0,
                       ca.if_else(e<0,
                       ca.if_else(-d<0, 
                       Q0P0_norm,0),0),0),0),0)
        ###2
        d2segment_18 = ca.if_else(parallel_flag <= self.ZERO,
                       ca.mmin(ca.horzcat(self.distance_point_to_segment(Q0,P0,P1,P0P1,P0P1_norm),
                                          self.distance_point_to_segment(Q1,P0,P1,P0P1,P0P1_norm))),0)
        
        d2segment=d2segment_1+d2segment_2+d2segment_3+d2segment_4+d2segment_5+d2segment_6+d2segment_7+d2segment_8+d2segment_9+\
                  d2segment_10+d2segment_11+d2segment_12+d2segment_13+d2segment_14+d2segment_15+d2segment_16+d2segment_17+d2segment_18
        """ f_d2segment_capsule = ca.Function('f_d2segment_capsule',[Q0,Q1,P0,P1],[d2segment])
        distance  =  f_d2segment_capsule(np.array([0,0,0]),np.array([1,0,0]),np.array([-1,0,0]),np.array([-2,-2,0]))
        print(distance) """
        
        return d2segment
    
    def distance_point_to_rectangle(self,point,RECTANGLE):
        RECTANGLE
        # E0=RECTANGLE.E0
        e0=RECTANGLE.e0
        U0=RECTANGLE.u0
        # E1=RECTANGLE.E1
        e1=RECTANGLE.e1
        U1=RECTANGLE.u1
        U2=RECTANGLE.u2
        CP=point-RECTANGLE.C
        #POINT = C + X0*U0+X1*U1+X2*U2
        X0=ca.dot(CP,U0)
        X1=ca.dot(CP,U1)
        X2=ca.dot(CP,U2)
        d2rectabgle_1 = ca.if_else(X0 > e0,
                        ca.if_else(X1 > e1,
                        (X0-e0)**2+(X1-e1)**2+X2**2,0),0)
        d2rectabgle_2 = ca.if_else(X0 > e0,
                        ca.if_else(-e1<= X1,
                        ca.if_else(X1 <= e1,
                        (X0-e0)**2+X2**2,0),0),0)
        d2rectabgle_3 = ca.if_else(X0 > e0,
                        ca.if_else(X1 < -e1,
                        (X0-e0)**2+(X1+e1)**2+X2**2,0),0)

        d2rectabgle_4 = ca.if_else(-e0<= X0,
                        ca.if_else(X0 <= e0,
                        ca.if_else(X1 > e1,
                        (X1-e1)**2+X2**2,0),0),0)
        d2rectabgle_5 = ca.if_else(-e0<= X0,
                        ca.if_else(X0 <= e0,
                        ca.if_else(-e1<= X1,
                        ca.if_else(X1 <= e1,
                        X2**2,0),0),0),0)
        d2rectabgle_6 = ca.if_else(-e0<= X0,
                        ca.if_else(X0 <= e0,
                        ca.if_else(X1 < -e1,
                        (X1+e1)**2+X2**2,0),0),0)

        d2rectabgle_7 = ca.if_else(X0 < -e0,
                        ca.if_else(X1 > e1,
                        (X0+e0)**2+(X1-e1)**2+X2**2,0),0)
        d2rectabgle_8 = ca.if_else(X0 < -e0,
                        ca.if_else(-e1<= X1,
                        ca.if_else(X1 <= e1,
                        (X0+e0)**2+X2**2,0),0),0)
        d2rectabgle_9 = ca.if_else(X0 < -e0,
                        ca.if_else(X1 < -e1,
                        (X0+e0)**2+(X1+e1)**2+X2**2,0),0)

        return ca.sqrt(d2rectabgle_1)+ca.sqrt(d2rectabgle_2)+ca.sqrt(d2rectabgle_3)\
              +ca.sqrt(d2rectabgle_4)+ca.sqrt(d2rectabgle_5)+ca.sqrt(d2rectabgle_6)\
              +ca.sqrt(d2rectabgle_7)+ca.sqrt(d2rectabgle_8)+ca.sqrt(d2rectabgle_9)

    def distance_segment_to_rectangle(self,RECTANGLE,P0,P1,P0P1,P0P1_norm): 
        P0P1 = P0P1 #P1-P0
        P0P1_norm = P0P1_norm #ca.norm_2(P0P1)
        C = RECTANGLE.C
        V0 = RECTANGLE.V0
        V1 = RECTANGLE.V1
        V2 = RECTANGLE.V2
        V3 = RECTANGLE.V3#V2+E0
        e0 = RECTANGLE.e0#ca.norm_2(E0)/2
        e1 = RECTANGLE.e1#ca.norm_2(E1)/2
        u0 = RECTANGLE.u0#E0/(2*e0)
        u1 = RECTANGLE.u1#E1/(2*e1)
        """ u2 = RECTANGLE.u2#ca.cross(u0,u1) 
        E0 = RECTANGLE.E0#V1-V0
        E1 = RECTANGLE.E1#V0-V2 """
        CP0 = P0-C
        CP1 = P1-C
        a0 = ca.dot(CP0,u0)
        b0 = ca.dot(CP0,u1)
        a1 = ca.dot(CP1,u0)
        b1 = ca.dot(CP1,u1)
        ### pre calculate for reusing
        a0Se0 = a0-e0
        a1Se0 = a1-e0
        a0Ae0 = a0+e0
        a1Ae0 = a1+e0
        b0Se1 = b0-e1
        b1Se1 = b1-e1
        b0Ae1 = b0+e1
        b1Ae1 = b1+e1
        b1Sb0 = b1-b0
        a1Sa0 = a1-a0
        PC0 = C+a0*u0+b0*u1
        PC1 = C+a1*u0+b1*u1
        P_e0=ca.if_else(a1Sa0!=0,C+e0*u0+((-a0Se0/a1Sa0)*b1Sb0+b0)*u1,0.000000001) 
        P_ne0=ca.if_else(a1Sa0!=0,C-e0*u0+((-a0Ae0/a1Sa0)*b1Sb0+b0)*u1,0.00000001) 
        P_e1=ca.if_else(b1Sb0!=0,C+((-b0Se1/b1Sb0)*a1Sa0+a0)*u0+e1*u1,0.00000001) 
        P_ne1=ca.if_else(b1Sb0!=0,C+((-b0Ae1/b1Sb0)*a1Sa0+a0)*u0-e1*u1,0.00000001)
        P_e0Ae1 = ca.if_else(a1Sa0!=0,(-a0Se0/a1Sa0)*b1Sb0+b0+e1,0.00000001)
        P_e0Se1 = ca.if_else(a1Sa0!=0,(-a0Se0/a1Sa0)*b1Sb0+b0-e1,0.00000001) 
        P_ne0Ae1 = ca.if_else(a1Sa0!=0,(-a0Ae0/a1Sa0)*b1Sb0+b0+e1,0.00000001)
        P_ne0Se1 = ca.if_else(a1Sa0!=0,(-a0Ae0/a1Sa0)*b1Sb0+b0-e1,0.00000001)
        P_e1Ae0 = ca.if_else(b1Sb0!=0,(-b0Se1/b1Sb0)*a1Sa0+a0+e0,0.00000001)
        P_e1Se0 = ca.if_else(b1Sb0!=0,(-b0Se1/b1Sb0)*a1Sa0+a0-e0,0.00000001) 
        P_ne1Ae0 = ca.if_else(b1Sb0!=0,(-b0Ae1/b1Sb0)*a1Sa0+a0+e0,0.00000001)
        P_ne1Se0 = ca.if_else(b1Sb0!=0,(-b0Ae1/b1Sb0)*a1Sa0+a0-e0,0.00000001)
        ###

        ###1.投影与矩形框无交点
        ##1.1
        S2R1=ca.if_else(ca.logic_and(a0Se0>0,a1Se0>0),self.distance_segment_to_segment(V1,V3,P0,P1,P0P1,P0P1_norm),0)
        ##1.2
        S2R2=ca.if_else(ca.logic_and(a0Ae0<0,a1Ae0<0),self.distance_segment_to_segment(V0,V2,P0,P1,P0P1,P0P1_norm),0)
        ##1.3
        S2R3=ca.if_else(ca.logic_or(ca.logic_or(ca.logic_and(ca.logic_and(b0Se1>0,b1Se1>0),ca.logic_and(a0Se0>0,a1Se0<=0)),
                                    ca.logic_and(ca.logic_and(b0Se1>0,b1Se1>0),ca.logic_and(a0Ae0>=0,a0Se0<=0))),
                                    ca.logic_and(ca.logic_and(b0Se1>0,b1Se1>0),ca.logic_and(a0Ae0<0,a1Ae0>=0)))
                        ,self.distance_segment_to_segment(V0,V1,P0,P1,P0P1,P0P1_norm),0)
        ##1.4
        S2R4=ca.if_else(ca.logic_or(ca.logic_or(ca.logic_and(ca.logic_and(b0Ae1<0,b1Ae1<0),ca.logic_and(a0Se0>0,a1Se0<=0)),
                                    ca.logic_and(ca.logic_and(b0Ae1<0,b1Ae1<0),ca.logic_and(a0Ae0>=0,a0Se0<=0))),
                                    ca.logic_and(ca.logic_and(b0Ae1<0,b1Se1<0),ca.logic_and(a0Ae0<0,a1Ae0>=0)))
                        ,self.distance_segment_to_segment(V2,V3,P0,P1,P0P1,P0P1_norm),0)
        ##1.5
        S2R5=ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a0Ae0>0,a0Se0<0),ca.logic_and(a1Ae0>0,a1Se0<0))
                                    ,ca.logic_and(ca.logic_and(b0Ae1>0,b0Se1<0),ca.logic_and(b1Ae1>0,b1Se1<0)))
                                    ,self.distance_segment_to_segment(PC0,PC1,P0,P1,P0P1,P0P1_norm),0)
        ##1.6
        S2R6=ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a0Se0*a1Se0<=0,b0Se1*b1Se1<=0),ca.logic_and(a1Sa0!=0,b1Sb0!=0))
                                    ,ca.logic_and(P_e0Se1>0,P_e1Se0>0))
                                    ,self.distance_point_to_segment(V1,P0,P1,P0P1,P0P1_norm),0)
        ##1.7
        S2R7=ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a0Ae0*a1Ae0<=0,b0Se1*b1Se1<=0),ca.logic_and(a1Sa0!=0,b1Sb0!=0))
                                    ,ca.logic_and(P_ne0Se1>0,P_e1Ae0<0))
                                    ,self.distance_point_to_segment(V0,P0,P1,P0P1,P0P1_norm),0)
        ##1.8
        S2R8=ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a0Ae0*a1Ae0<=0,b0Ae1*b1Ae1<=0),ca.logic_and(a1Sa0!=0,b1Sb0!=0))
                                    ,ca.logic_and(P_ne0Ae1<0,P_ne1Ae0<0))
                                    ,self.distance_point_to_segment(V2,P0,P1,P0P1,P0P1_norm),0)
        ##1.9
        S2R9=ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a0Se0*a1Se0<=0,b0Ae1*b1Ae1<=0),ca.logic_and(a1Sa0!=0,b1Sb0!=0))
                                    ,ca.logic_and(P_e0Ae1<0,P_ne1Se0>0))
                                    ,self.distance_point_to_segment(V3,P0,P1,P0P1,P0P1_norm),0)
        S2R_SUM1 = S2R1+S2R2+S2R3+S2R4+S2R5+S2R6+S2R7+S2R8+S2R9
        ###2.投影与矩形框有1个交点
        ##2.1 V0_P,V1_P
        #2.1.1
        S2R10_1 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(b1Se1>=0,b1Sb0>0)
        ,ca.logic_and(P_e1Se0<0,P_e1Ae0>0))
        ,ca.logic_and(ca.logic_and(a0Ae0>0,a0Se0<0),ca.logic_and(b0Ae1>0,b0Se1<=0)))
        ,self.distance_segment_to_segment(P_e1,PC0,P0,P1,P0P1,P0P1_norm),0)
        #2.1.2
        S2R10_2 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(b0Se1>=0,b1Sb0<0)
        ,ca.logic_and(P_e1Se0<0,P_e1Ae0>0))
        ,ca.logic_and(ca.logic_and(a1Ae0>0,a1Se0<0),ca.logic_and(b1Ae1>0,b1Se1<=0)))
        ,self.distance_segment_to_segment(P_e1,PC1,P0,P1,P0P1,P0P1_norm),0)
        #2.1.3
        S2R10_3 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a1Sa0==0,b1Sb0==0),b1Se1==0),ca.logic_and(a0Ae0>0,a0Se0<0))
                            ,self.distance_point_to_segment(PC0,P0,P1,P0P1,P0P1_norm),0)

        S2R10 = S2R10_1+S2R10_2+S2R10_3
        ##2.2 V0_P,V2_P
        #2.2.1
        S2R11_1 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a1Ae0<=0,a1Sa0<0)
        ,ca.logic_and(P_ne0Se1<0,P_ne0Ae1>0))
        ,ca.logic_and(ca.logic_and(a0Ae0>=0,a0Se0<0),ca.logic_and(b0Ae1>0,b0Se1<0)))
        ,self.distance_segment_to_segment(P_ne0,PC0,P0,P1,P0P1,P0P1_norm),0)
        #2.2.2
        S2R11_2 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a0Ae0<=0,a1Sa0>0)
        ,ca.logic_and(P_ne0Se1<0,P_ne0Ae1>0))
        ,ca.logic_and(ca.logic_and(a1Ae0>=0,a1Se0<0),ca.logic_and(b1Ae1>0,b1Se1<0)))
        ,self.distance_segment_to_segment(P_ne0,PC1,P0,P1,P0P1,P0P1_norm),0)
        #2.2.3
        S2R11_3 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a1Sa0==0,b1Sb0==0),a1Ae0==0),ca.logic_and(b0Ae1>0,b0Se1<0))
                            ,self.distance_point_to_segment(PC0,P0,P1,P0P1,P0P1_norm),0)
        S2R11 = S2R11_1+S2R11_2+S2R11_3
        ##2.3 V2_P,V3_P
        #2.3.1
        S2R12_1 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(b1Ae1<=0,b1Sb0<0)
        ,ca.logic_and(P_ne1Se0<0,P_ne1Ae0>0))
        ,ca.logic_and(ca.logic_and(a0Ae0>0,a0Se0<0),ca.logic_and(b0Ae1>=0,b0Se1<0)))
        ,self.distance_segment_to_segment(P_ne1,PC0,P0,P1,P0P1,P0P1_norm),0) 
        #2.3.2
        S2R12_2 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(b0Ae1<=0,b1Sb0>0)
        ,ca.logic_and(P_ne1Se0<0,P_ne1Ae0>0))
        ,ca.logic_and(ca.logic_and(a1Ae0>0,a1Se0<0),ca.logic_and(b1Ae1>=0,b1Se1<0)))
        ,self.distance_segment_to_segment(P_ne1,PC1,P0,P1,P0P1,P0P1_norm),0)
        #2.3.3
        S2R12_3 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a1Sa0==0,b1Sb0==0),b1Ae1==0),ca.logic_and(a0Ae0>0,a0Se0<0))
                            ,self.distance_point_to_segment(PC0,P0,P1,P0P1,P0P1_norm),0)
        S2R12 = S2R12_1+S2R12_2+S2R12_3
        ##2.4 V3_P,V1_P
        #2.4.1
        S2R13_1 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a1Se0>=0,a1Sa0>0)
        ,ca.logic_and(P_e0Se1<0,P_e0Ae1>0))
        ,ca.logic_and(ca.logic_and(a0Ae0>0,a0Se0<=0),ca.logic_and(b0Ae1>0,b0Se1<0)))
        ,self.distance_segment_to_segment(P_e0,PC0,P0,P1,P0P1,P0P1_norm),0)
        #2.4.2
        S2R13_2 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a0Se0>=0,a1Sa0<0)
        ,ca.logic_and(P_e0Se1<0,P_e0Ae1>0))
        ,ca.logic_and(ca.logic_and(a1Ae0>0,a1Se0<=0),ca.logic_and(b1Ae1>0,b1Se1<0)))
        ,self.distance_segment_to_segment(P_e0,PC1,P0,P1,P0P1,P0P1_norm),0)
        #2.4.3
        S2R13_3 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a1Sa0==0,b1Sb0==0),a1Se0==0),ca.logic_and(b0Ae1>0,b0Se1<0))
                            ,self.distance_point_to_segment(PC0,P0,P1,P0P1,P0P1_norm),0)    
        S2R13 = S2R13_1+S2R13_2+S2R13_3
        ##2.5 V1_P
        #2.5.1
        S2R14_1 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(ca.logic_and(a0Ae0>0,a0Se0<0),ca.logic_and(b0Ae1>0,b0Se1<0))
                                                    ,ca.logic_and(b1Se1>=0,a1Se0>=0))
                                        ,(b1Sb0/a1Sa0)==(b0Se1/a0Se0)),
                            self.distance_segment_to_segment(PC0,V1,P0,P1,P0P1,P0P1_norm),0)
        #2.5.3                  
        S2R14_2 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(ca.logic_and(a1Ae0>0,a1Se0<0),ca.logic_and(b1Ae1>0,b1Se1<0))
                                                    ,ca.logic_and(b0Se1>=0,a0Se0>=0))
                                        ,(b1Sb0/a1Sa0)==(b1Se1/a1Se0)),
                            self.distance_segment_to_segment(PC1,V1,P0,P1,P0P1,P0P1_norm),0)      
        #2.5.2;2.5.4;2.5.5
        S2R14_3 = ca.if_else(ca.logic_or(ca.logic_or(ca.logic_and(ca.logic_and(a0Se0==0,b0Se1==0),ca.logic_or(a1Se0>0,b1Se1>0))
                                                    ,ca.logic_and(ca.logic_and(a1Se0==0,b1Se1==0),ca.logic_or(a0Se0>0,b0Se1>0))),
                                        ca.logic_and(ca.logic_and(a1Sa0==0,a0Se0==0),ca.logic_and(b1Sb0==0,b0Se1==0)))
                            ,self.distance_point_to_segment(V1,P0,P1,P0P1,P0P1_norm),0)
        S2R14 = S2R14_1+S2R14_2+S2R14_3
        ##2.6 V0_P
        #2.6.1
        S2R15_1 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(ca.logic_and(a0Ae0>0,a0Se0<0),ca.logic_and(b0Ae1>0,b0Se1<0))
                                                    ,ca.logic_and(b1Se1>=0,a1Ae0<=0))
                                        ,(b1Sb0/a1Sa0)==(b0Se1/a0Ae0)),
                            self.distance_segment_to_segment(PC0,V0,P0,P1,P0P1,P0P1_norm),0)
        #2.6.3                  
        S2R15_2 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(ca.logic_and(a1Ae0>0,a1Se0<0),ca.logic_and(b1Ae1>0,b1Se1<0))
                                                    ,ca.logic_and(b0Se1>=0,a0Ae0<=0))
                                        ,(b1Sb0/a1Sa0)==(b1Se1/a1Ae0)),
                            self.distance_segment_to_segment(PC1,V0,P0,P1,P0P1,P0P1_norm),0)      
        #2.6.2;2.6.4;2.6.5
        S2R15_3 = ca.if_else(ca.logic_or(ca.logic_or(ca.logic_and(ca.logic_and(a0Ae0==0,b0Se1==0),ca.logic_or(a1Ae0<0,b1Se1>0))
                                                    ,ca.logic_and(ca.logic_and(a1Ae0==0,b1Se1==0),ca.logic_or(a0Ae0<0,b0Se1>0))),
                                        ca.logic_and(ca.logic_and(a1Sa0==0,a0Ae0==0),ca.logic_and(b1Sb0==0,b0Se1==0)))
                            ,self.distance_point_to_segment(V0,P0,P1,P0P1,P0P1_norm),0)
        S2R15 = S2R15_1+S2R15_2+S2R15_3
        ##2.7 V2_P
        #2.7.1
        S2R16_1 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(ca.logic_and(a0Ae0>0,a0Se0<0),ca.logic_and(b0Ae1>0,b0Se1<0))
                                                    ,ca.logic_and(b1Ae1<=0,a1Ae0<=0))
                                        ,(b1Sb0/a1Sa0)==(b0Ae1/a0Ae0)),
                            self.distance_segment_to_segment(PC0,V2,P0,P1,P0P1,P0P1_norm),0)
        #2.7.3                  
        S2R16_2 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(ca.logic_and(a1Ae0>0,a1Se0<0),ca.logic_and(b1Ae1>0,b1Se1<0))
                                                    ,ca.logic_and(b0Ae1<=0,a0Ae0<=0))
                                        ,(b1Sb0/a1Sa0)==(b1Ae1/a1Ae0)),
                            self.distance_segment_to_segment(PC1,V2,P0,P1,P0P1,P0P1_norm),0)      
        #2.7.2;2.7.4;2.7.5
        S2R16_3 = ca.if_else(ca.logic_or(ca.logic_or(ca.logic_and(ca.logic_and(a0Ae0==0,b0Ae1==0),ca.logic_or(a1Ae0<0,b1Ae1<0))
                                                    ,ca.logic_and(ca.logic_and(a1Ae0==0,b1Ae1==0),ca.logic_or(a0Ae0<0,b0Ae1<0))),
                                        ca.logic_and(ca.logic_and(a1Sa0==0,a0Ae0==0),ca.logic_and(b1Sb0==0,b0Ae1==0)))
                            ,self.distance_point_to_segment(V2,P0,P1,P0P1,P0P1_norm),0)
        S2R16 = S2R16_1+S2R16_2+S2R16_3
        ##2.8 V3_P
        #2.8.1
        S2R17_1 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(ca.logic_and(a0Ae0>0,a0Se0<0),ca.logic_and(b0Ae1>0,b0Se1<0))
                                                    ,ca.logic_and(b1Ae1<=0,a1Se0>=0))
                                        ,(b1Sb0/a1Sa0)==(b0Ae1/a0Se0)),
                            self.distance_segment_to_segment(PC0,V3,P0,P1,P0P1,P0P1_norm),0)
        #2.8.3                  
        S2R17_2 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(ca.logic_and(a1Ae0>0,a1Se0<0),ca.logic_and(b1Ae1>0,b1Se1<0))
                                                    ,ca.logic_and(b0Ae1<=0,a0Se0>=0))
                                        ,(b1Sb0/a1Sa0)==(b1Ae1/a1Se0)),
                            self.distance_segment_to_segment(PC1,V3,P0,P1,P0P1,P0P1_norm),0)      
        #2.8.2;2.8.4;2.8.5
        S2R17_3 = ca.if_else(ca.logic_or(ca.logic_or(ca.logic_and(ca.logic_and(a0Se0==0,b0Ae1==0),ca.logic_or(a1Se0>0,b1Ae1<0))
                                                    ,ca.logic_and(ca.logic_and(a1Se0==0,b1Ae1==0),ca.logic_or(a0Se0>0,b0Ae1<0))),
                                        ca.logic_and(ca.logic_and(a1Sa0==0,a0Se0==0),ca.logic_and(b1Sb0==0,b0Ae1==0)))
                            ,self.distance_point_to_segment(V3,P0,P1,P0P1,P0P1_norm),0)
        S2R17 = S2R17_1+S2R17_2+S2R17_3
        S2R_SUM2 = S2R10+S2R11+S2R12+S2R13+S2R14+S2R15+S2R16+S2R17
        ###3.投影与矩形框有2个交点
        ##3.1 交点在对边
        #3.1.1 左右  
        S2R18_1 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a0Se0*a1Se0<=0,a0Ae0*a1Ae0<=0),a1Sa0!=0)
                                        ,ca.logic_and(ca.logic_and(P_e0Se1<0,P_e0Ae1>0),ca.logic_and(P_ne0Se1<0,P_ne0Ae1>0)))
                            ,self.distance_segment_to_segment(P_e0,P_ne0,P0,P1,P0P1,P0P1_norm),0)
        #3.1.2 上下
        S2R18_2 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(b0Se1*b1Se1<=0,b0Ae1*b1Ae1<=0),b1Sb0!=0)
                                        ,ca.logic_and(ca.logic_and(P_e1Se0<0,P_e1Ae0>0),ca.logic_and(P_ne1Se0<0,P_ne1Ae0>0)))
                            ,self.distance_segment_to_segment(P_e1,P_ne1,P0,P1,P0P1,P0P1_norm),0)
        S2R18 = S2R18_1+S2R18_2
        ##3.2 交点在邻边
        #3.2.1 (2 4 5)
        S2R19_1 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a1Sa0!=0,b1Sb0!=0),ca.logic_and(a0Se0*a1Se0<=0,b0Se1*b1Se1<=0))
                            ,ca.logic_and(ca.logic_and(P_e0Se1<0,P_e0Ae1>=0),ca.logic_and(P_e1Se0<0,P_e1Ae0>=0)))
                            ,self.distance_segment_to_segment(P_e0,P_e1,P0,P1,P0P1,P0P1_norm),0)
        #3.2.2 (4 5 8)
        S2R19_2 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(a1Sa0!=0,b1Sb0!=0),ca.logic_and(a0Ae0*a1Ae0<=0,b0Se1*b1Se1<=0))
                            ,ca.logic_and(ca.logic_and(P_ne0Se1<0,P_ne0Ae1>=0),ca.logic_and(P_e1Se0<=0,P_e1Ae0>0)))
                            ,self.distance_segment_to_segment(P_ne0,P_e1,P0,P1,P0P1,P0P1_norm),0)
        #3.2.3 (5 6 8)
        S2R19_3 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(ca.logic_and(a1Sa0!=0,b1Sb0!=0),ca.logic_and(a0Ae0*a1Ae0<=0,b0Ae1*b1Ae1<=0))
                                                    ,ca.logic_and(ca.logic_and(P_ne0Se1<=0,P_ne0Ae1>0),ca.logic_and(P_ne1Se0<=0,P_ne1Ae0>0)))
                                        ,(P_ne0Se1+P_ne1Se0)!=0)
                            ,self.distance_segment_to_segment(P_ne0,P_ne1,P0,P1,P0P1,P0P1_norm),0)
        #3.2.4 (2 5 6)
        S2R19_4 = ca.if_else(ca.logic_and(ca.logic_and(ca.logic_and(ca.logic_and(a1Sa0!=0,b1Sb0!=0),ca.logic_and(a0Se0*a1Se0<=0,b0Ae1*b1Ae1<=0))
                                                    ,ca.logic_and(ca.logic_and(P_e0Se1<=0,P_e0Ae1>0),ca.logic_and(P_ne1Se0<0,P_ne1Ae0>=0)))
                                        ,(P_e0Se1+P_ne1Ae0)!=0)
                            ,self.distance_segment_to_segment(P_e0,P_ne1,P0,P1,P0P1,P0P1_norm),0)

        S2R19 = S2R19_1+S2R19_2+S2R19_3+S2R19_4

        S2R_SUM3 = S2R18+S2R19
        ###4.投影与矩形框有无穷个交点
        ## 4.1与V1_P,V3_P重合
        S2R20 = ca.if_else(ca.logic_or(ca.logic_or(ca.logic_and(ca.logic_and(a1Sa0==0,a0Se0==0),ca.logic_and(b0Se1>=0,b1Se1<0))
                                                ,ca.logic_and(ca.logic_and(ca.logic_and(a1Sa0==0,a0Se0==0),ca.logic_and(b0Se1<0,b0Ae1>0))
                                                            ,b1Sb0!=0))
                                    ,ca.logic_and(ca.logic_and(a1Sa0==0,a0Se0==0),ca.logic_and(b0Ae1<=0,b1Ae1>0)))
                    ,self.distance_segment_to_segment(V1,V3,P0,P1,P0P1,P0P1_norm),0)
        
        ## 4.2与V1_P,V0_P重合
        S2R21 = ca.if_else(ca.logic_or(ca.logic_or(ca.logic_and(ca.logic_and(b1Sb0==0,b0Se1==0),ca.logic_and(a0Ae0<=0,a1Ae0>0))
                                                ,ca.logic_and(ca.logic_and(ca.logic_and(b1Sb0==0,b0Se1==0),ca.logic_and(a0Se0<0,a0Ae0>0))
                                                            ,a1Sa0!=0))
                                    ,ca.logic_and(ca.logic_and(b1Sb0==0,b0Se1==0),ca.logic_and(a0Se0>=0,a1Se0<0)))
                    ,self.distance_segment_to_segment(V0,V1,P0,P1,P0P1,P0P1_norm),0)      
        ## 4.3与V0_P,V2_P重合
        S2R22 = ca.if_else(ca.logic_or(ca.logic_or(ca.logic_and(ca.logic_and(a1Sa0==0,a0Ae0==0),ca.logic_and(b0Se1>=0,b1Se1<0))
                                                ,ca.logic_and(ca.logic_and(ca.logic_and(a1Sa0==0,a0Ae0==0),ca.logic_and(b0Se1<0,b0Ae1>0))
                                                            ,b1Sb0!=0))
                                    ,ca.logic_and(ca.logic_and(a1Sa0==0,a0Ae0==0),ca.logic_and(b0Ae1<=0,b1Ae1>0)))
                    ,self.distance_segment_to_segment(V0,V2,P0,P1,P0P1,P0P1_norm),0)  
        ## 4.4与V2_P,V3_P重合
        S2R23 = ca.if_else(ca.logic_or(ca.logic_or(ca.logic_and(ca.logic_and(b1Sb0==0,b0Ae1==0),ca.logic_and(a0Ae0<=0,a1Ae0>0))
                                                ,ca.logic_and(ca.logic_and(ca.logic_and(b1Sb0==0,b0Ae1==0),ca.logic_and(a0Se0<0,a0Ae0>0))
                                                            ,a1Sa0!=0))
                                    ,ca.logic_and(ca.logic_and(b1Sb0==0,b0Ae1==0),ca.logic_and(a0Se0>=0,a1Se0<0)))
                    ,self.distance_segment_to_segment(V2,V3,P0,P1,P0P1,P0P1_norm),0) 

        S2R_SUM4 = S2R20+S2R21+S2R22+S2R23
        S2R=S2R_SUM1+S2R_SUM2+S2R_SUM3+S2R_SUM4
        """ print(f"S2R_SUM1={S2R_SUM1}")
        print(f"S2R_SUM2={S2R_SUM2}")
        print(f"S2R_SUM3={S2R_SUM3}")
        print(f"S2R_SUM4={S2R_SUM4}") """
        return S2R

    def initial_sphere_obstacle_CBF(self,Obstacle,safe_R,Obs_v,i):
        i=i*self.n_conpoment
        gama = self.gama
        pre_Obstacle=Obstacle+Obs_v.T*self.T_step
        ### distance of obstacle to arm segment
        #BASEA
        d2segment_BASEA = self.distance_point_to_segment(pre_Obstacle,self.BASE,self.A,self.BASEA,self.BASEA_norm)
        #AB
        d2segment_AB = self.distance_point_to_segment(pre_Obstacle,self.A,self.B,self.AB,self.AB_norm)
        #BC
        d2segment_BC = self.distance_point_to_segment(pre_Obstacle,self.B,self.C,self.BC,self.BC_norm)
        #CD
        d2segment_CD = self.distance_point_to_segment(pre_Obstacle,self.C,self.D,self.CD,self.CD_norm)
        #DE
        d2segment_DE = self.distance_point_to_segment(pre_Obstacle,self.D,self.E,self.DE,self.DE_norm)
        #EEND
        d2segment_EEND = self.distance_point_to_segment(pre_Obstacle,self.E,self.END,self.EEND,self.EEND_norm)
        #BASE
        d2segment_BASE = self.distance_point_to_point(pre_Obstacle,self.BASE)

        self.dis2obs[i+0] =   d2segment_BASEA
        self.dis2obs[i+1] =   d2segment_AB
        self.dis2obs[i+2] =   d2segment_BC
        self.dis2obs[i+3] =   d2segment_CD
        self.dis2obs[i+4] =   d2segment_DE
        self.dis2obs[i+5] =   d2segment_EEND
        self.dis2obs[i+6] =   d2segment_BASE
        Obs_v_horzcat = Obs_v.T
        ### CBF of arm   0.12 
        for j in range(self.n_conpoment):
            ### CBF of arm
            locals()[f'h{j}']=self.dis2obs[i+j]**2 - safe_R[j]**2
            self.h[i+j]=locals()[f'h{j}']
            ###DERIVATIVE ALONG ANGLES
            locals()[f'dh{j}dx']=ca.jacobian(locals()[f'h{j}'],self.state)
            self.dhdx[i+j,:]=locals()[f'dh{j}dx']
            ###DERIVATIVE ALONG obs_p
            locals()[f'dh{j}dp']=ca.jacobian(locals()[f'h{j}'],Obstacle)
            # self.dhdp[i+j,:] = locals()[f'dh{j}dp']
            ### Lghx = G = dhdx @ gx
            locals()[f'G{j}']=locals()[f'dh{j}dx'] @ self.gx
            self.G[i+j,:]=locals()[f'G{j}']
            ### lg2hx = K = dhdx @ g2
            locals()[f'K{j}']=locals()[f'dh{j}dx'] @ self.g2
            self.K[i+j,:]=locals()[f'K{j}']

            if self.CBF_mode=='11':
                # H0 = dh0dx@self.Ax + gama * h0 - self.param1[i+0] - self.param2[i+0] * (ca.norm_2(K0)**2) + dh0dp@Obs_v.T - ca.norm_2(dh0dp)*self.sigma_v
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat - ca.norm_2(locals()[f'dh{j}dp'])*self.sigma_v
                
            if self.CBF_mode=='10':
                # H0 = dh0dx@self.Ax + gama * h0 - self.param1[i+0] - self.param2[i+0] * (ca.norm_2(K0)**2) + dh0dp@Obs_v.T 
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat
                
            if self.CBF_mode=='01':
                # H0 = dh0dx@self.Ax + gama * h0 + dh0dp@Obs_v.T - ca.norm_2(dh0dp)*self.sigma_v
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat - ca.norm_2(locals()[f'dh{j}dp'])*self.sigma_v
                
            if self.CBF_mode=='00':
                # H0 = dh0dx@self.Ax + gama * h0 + dh0dp@Obs_v.T 
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat

            self.H[i+j] = locals()[f'H{j}']

    def initial_capsule_obstacle_CBF(self,Obstacle,safe_R,Obs_v,i):
        i=(i+self.sphere_obstacle_num)*self.n_conpoment
        gama=0.9*self.gama
        ### distance of obstacle to arm line  Obstacle:segment(Q0,Q1)
        pre_Q0=Obstacle[0:3]+Obs_v.T*self.T_step
        pre_Q1=Obstacle[3:6]+Obs_v.T*self.T_step
        #BASEA
        d2segment_BASEA = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.BASE,self.A,self.BASEA,self.BASEA_norm)
        #AB
        d2segment_AB = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.A,self.B,self.AB,self.AB_norm)
        #BC1
        d2segment_BC = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.B,self.C,self.BC,self.BC_norm)
        #C0C1
        d2segment_CD = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.C,self.D,self.CD,self.CD_norm)
        #C0D
        d2segment_DE = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.D,self.E,self.DE,self.DE_norm)
        #DE
        d2segment_EEND = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.E,self.END,self.EEND,self.EEND_norm)               
        #BASE
        pre_O0O1=pre_Q1-pre_Q0
        d2segment_BASE = self.distance_point_to_segment(self.BASE,pre_Q0,pre_Q1,pre_O0O1,ca.norm_2(pre_O0O1))

        self.dis2obs[i+0] =   d2segment_BASEA
        self.dis2obs[i+1] =   d2segment_AB
        self.dis2obs[i+2] =   d2segment_BC  
        self.dis2obs[i+3] =   d2segment_CD
        self.dis2obs[i+4] =   d2segment_DE
        self.dis2obs[i+5] =   d2segment_EEND
        self.dis2obs[i+6] =   d2segment_BASE
        Obs_v_horzcat=ca.horzcat(Obs_v,Obs_v).T
        ### CBF of arm   0.12 
        for j in range(self.n_conpoment):
            ### CBF of arm
            locals()[f'h{j}']=self.dis2obs[i+j]**2 - safe_R[j]**2
            self.h[i+j]=locals()[f'h{j}']
            ###DERIVATIVE ALONG ANGLES
            locals()[f'dh{j}dx']=ca.jacobian(locals()[f'h{j}'],self.state)
            self.dhdx[i+j,:]=locals()[f'dh{j}dx']
            ###DERIVATIVE ALONG obs_p
            locals()[f'dh{j}dp']=ca.jacobian(locals()[f'h{j}'],Obstacle)
            # self.dhdp[i+j,:] = locals()[f'dh{j}dp']
            ### Lghx = G = dhdx @ gx
            locals()[f'G{j}']=locals()[f'dh{j}dx'] @ self.gx
            self.G[i+j,:]=locals()[f'G{j}']
            ### lg2hx = K = dhdx @ g2
            locals()[f'K{j}']=locals()[f'dh{j}dx'] @ self.g2
            self.K[i+j,:]=locals()[f'K{j}']

            if self.CBF_mode=='11':
                # H0 = dh0dx@self.Ax + gama * h0 - self.param1[i+0] - self.param2[i+0] * (ca.norm_2(K0)**2) + dh0dp@Obs_v.T - ca.norm_2(dh0dp)*self.sigma_v
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat - ca.norm_2(locals()[f'dh{j}dp'])*self.sigma_v
                
            if self.CBF_mode=='10':
                # H0 = dh0dx@self.Ax + gama * h0 - self.param1[i+0] - self.param2[i+0] * (ca.norm_2(K0)**2) + dh0dp@Obs_v.T 
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat
                
            if self.CBF_mode=='01':
                # H0 = dh0dx@self.Ax + gama * h0 + dh0dp@Obs_v.T - ca.norm_2(dh0dp)*self.sigma_v
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat - ca.norm_2(locals()[f'dh{j}dp'])*self.sigma_v
                
            if self.CBF_mode=='00':
                # H0 = dh0dx@self.Ax + gama * h0 + dh0dp@Obs_v.T 
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat

            self.H[i+j] = locals()[f'H{j}']
 
    def initial_rectangle_obstacle_CBF(self,Obstacle,safe_R,Obs_v,i):
        i=(i+self.sphere_obstacle_num+self.capsule_obstacle_num)*self.n_conpoment
        gama=0.8*self.gama

        ### distance of obstacle to arm line Obstacle:rectangle(C,V0,V1,V2)
        rectangle = RECTANGLE(Obstacle[0:3],Obstacle[3:6],Obstacle[6:9],Obstacle[9:12])
        pre_rectangle=RECTANGLE(Obstacle[0:3]+Obs_v.T*self.T_step,Obstacle[3:6]+Obs_v.T*self.T_step,Obstacle[6:9]+Obs_v.T*self.T_step,Obstacle[9:12]+Obs_v.T*self.T_step)
        #BASEA
        d2segment_BASEA = self.distance_segment_to_rectangle(pre_rectangle,self.BASE,self.A,self.BASEA,self.BASEA_norm)
        #AB
        d2segment_AB = self.distance_segment_to_rectangle(pre_rectangle,self.A,self.B,self.AB,self.AB_norm)
        #BC1
        d2segment_BC = self.distance_segment_to_rectangle(pre_rectangle,self.B,self.C,self.BC,self.BC_norm)
        #C0C1
        d2segment_CD = self.distance_segment_to_rectangle(pre_rectangle,self.C,self.D,self.CD,self.CD_norm)
        #C0D
        d2segment_DE = self.distance_segment_to_rectangle(pre_rectangle,self.D,self.E,self.DE,self.DE_norm)
        #DE
        d2segment_EEND = self.distance_segment_to_rectangle(pre_rectangle,self.E,self.END,self.EEND,self.EEND_norm)
        #BASE
        d2segment_BASE = self.distance_point_to_rectangle(self.BASE,pre_rectangle)

        self.dis2obs[i+0] =   d2segment_BASEA
        self.dis2obs[i+1] =   d2segment_AB
        self.dis2obs[i+2] =   d2segment_BC
        self.dis2obs[i+3] =   d2segment_CD
        self.dis2obs[i+4] =   d2segment_DE
        self.dis2obs[i+5] =   d2segment_EEND
        self.dis2obs[i+6] =   d2segment_BASE
        Obs_v_horzcat=ca.horzcat(Obs_v,Obs_v,Obs_v,Obs_v).T
        ### CBF of arm   0.12 
        for j in range(self.n_conpoment):
            ### CBF of arm
            locals()[f'h{j}']=self.dis2obs[i+j]**2 - safe_R[j]**2 #2
            self.h[i+j]=locals()[f'h{j}']
            ###DERIVATIVE ALONG ANGLES
            locals()[f'dh{j}dx']=ca.jacobian(locals()[f'h{j}'],self.state)
            self.dhdx[i+j,:]=locals()[f'dh{j}dx']
            ###DERIVATIVE ALONG obs_p
            locals()[f'dh{j}dp']=ca.jacobian(locals()[f'h{j}'],Obstacle)
            # self.dhdp[i+j,:] = locals()[f'dh{j}dp']
            ### Lghx = G = dhdx @ gx
            locals()[f'G{j}']=locals()[f'dh{j}dx'] @ self.gx
            self.G[i+j,:]=locals()[f'G{j}']
            ### lg2hx = K = dhdx @ g2
            locals()[f'K{j}']=locals()[f'dh{j}dx'] @ self.g2
            self.K[i+j,:]=locals()[f'K{j}']

            if self.CBF_mode=='11':
                # H0 = dh0dx@self.Ax + gama * h0 - self.param1[i+0] - self.param2[i+0] * (ca.norm_2(K0)**2) + dh0dp@Obs_v.T - ca.norm_2(dh0dp)*self.sigma_v
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat - ca.norm_2(locals()[f'dh{j}dp'])*self.sigma_v
                
            if self.CBF_mode=='10':
                # H0 = dh0dx@self.Ax + gama * h0 - self.param1[i+0] - self.param2[i+0] * (ca.norm_2(K0)**2) + dh0dp@Obs_v.T 
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat
                
            if self.CBF_mode=='01':
                # H0 = dh0dx@self.Ax + gama * h0 + dh0dp@Obs_v.T - ca.norm_2(dh0dp)*self.sigma_v
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat - ca.norm_2(locals()[f'dh{j}dp'])*self.sigma_v
                
            if self.CBF_mode=='00':
                # H0 = dh0dx@self.Ax + gama * h0 + dh0dp@Obs_v.T 
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                + locals()[f'dh{j}dp']@Obs_v_horzcat

            self.H[i+j] = locals()[f'H{j}']

    def initial_base_CBF(self):
        i=self.totle_num*self.n_conpoment
        # Obstacle = self.BASE.T
        Obstacle = self.BASE
        safe_R_table = self.R_safe_base
        gama=1.5*self.gama
        ### distance of obstacle to arm segment
        #CD
        d2segment_CD = self.distance_point_to_segment(Obstacle,self.C,self.D,self.DE,self.CD_norm)
        #DE
        d2segment_DE = self.distance_point_to_segment(Obstacle,self.D,self.E,self.DE,self.DE_norm)
        #EEND
        d2segment_EEND = self.distance_point_to_segment(Obstacle,self.E,self.END,self.EEND,self.EEND_norm)

        self.dis2obs[i+0] =   d2segment_CD
        self.dis2obs[i+1] =   d2segment_DE
        self.dis2obs[i+2] =   d2segment_EEND

        for j in range(self.n_base_cbf):
            ### CBF of arm
            locals()[f'h{j}']=self.dis2obs[i+j]**2 - safe_R_table**2
            self.h[i+j]=locals()[f'h{j}']
            ###DERIVATIVE ALONG ANGLES
            locals()[f'dh{j}dx']=ca.jacobian(locals()[f'h{j}'],self.state)
            self.dhdx[i+j,:]=locals()[f'dh{j}dx']
            ### Lghx = G = dhdx @ gx
            locals()[f'G{j}']=locals()[f'dh{j}dx'] @ self.gx
            self.G[i+j,:]=locals()[f'G{j}']
            ### lg2hx = K = dhdx @ g2
            locals()[f'K{j}']=locals()[f'dh{j}dx'] @ self.g2
            self.K[i+j,:]=locals()[f'K{j}']
            if self.CBF_mode=='11':
                # H2 = dh2dx@self.Ax + gama * h2 - self.param1[i+2] - self.param2[i+2] * (ca.norm_2(K2)**2) 
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) 
                               
            if self.CBF_mode=='10':
                # H2 = dh2dx@self.Ax + gama * h2 - self.param1[i+2] - self.param2[i+2] * (ca.norm_2(K2)**2) 
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) 
                
            if self.CBF_mode=='01':
                # H2 = dh2dx@self.Ax + gama * h2
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] 
                
            if self.CBF_mode=='00':
                # H2 = dh2dx@self.Ax + gama * h2
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] 

            self.H[i+j] = locals()[f'H{j}']
        
    def initial_statistic_CBF(self):
        #高于地面
        gama=self.gama
        i=self.totle_num*self.n_conpoment+self.n_base_cbf
        h0 = self.C[2]+0.052 - 0.08
        h1 = self.D[2]+0.052 - 0.08
        h2 = self.E[2]+0.052 - 0.08
        h3 = self.END[2]+0.052 - 0.08
        h4 = -self.base_posture[0] + 10.0
        h5 = self.base_posture[0] + 10.0
        h6 = -self.base_posture[1] + 10.0
        h7 = self.base_posture[1] + 10.0
        
        #活动范围 ：高于地面以及在矩形内
        for j in range(self.n_env_limite_cbf):
            self.h[i+j]=locals()[f'h{j}']
            ###DERIVATIVE ALONG ANGLES
            locals()[f'dh{j}dx']=ca.jacobian(locals()[f'h{j}'],self.state)
            self.dhdx[i+j,:]=locals()[f'dh{j}dx']
            ### Lghx = G = dhdx @ gx
            locals()[f'G{j}']=locals()[f'dh{j}dx'] @ self.gx
            self.G[i+j,:]=locals()[f'G{j}']
            ### lg2hx = K = dhdx @ g2
            locals()[f'K{j}']=locals()[f'dh{j}dx'] @ self.g2
            self.K[i+j,:]=locals()[f'K{j}']
            if self.CBF_mode=='11':
                # H0 = dh0dx@self.Ax + gama * h0 - self.param1[i+0] - self.param2[i+0] * (ca.norm_2(K0)**2) + dh0dp@Obs_v.T - ca.norm_2(dh0dp)*self.sigma_v
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) 
                
            if self.CBF_mode=='10':
                # H0 = dh0dx@self.Ax + gama * h0 - self.param1[i+0] - self.param2[i+0] * (ca.norm_2(K0)**2) + dh0dp@Obs_v.T 
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) 
                
            if self.CBF_mode=='01':
                # H0 = dh0dx@self.Ax + gama * h0 + dh0dp@Obs_v.T - ca.norm_2(dh0dp)*self.sigma_v
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] 
                
            if self.CBF_mode=='00':
                # H0 = dh0dx@self.Ax + gama * h0 + dh0dp@Obs_v.T 
                locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] 

            self.H[i+j] = locals()[f'H{j}']
        #静止障碍物：四个胶囊障碍物
        for k in range(self.statistic_capsule_obstacle_num):
            Obstacle = self.statistic_capsule_obstacle_list[k,:]
            i = self.totle_num*self.n_conpoment+self.n_base_cbf+self.n_env_limite_cbf+k*self.n_conpoment
            gama=0.5*self.gama
            pre_Q0=Obstacle[0:3].T
            pre_Q1=Obstacle[3:6].T
            #BASEA
            d2segment_BASEA = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.BASE,self.A,self.BASEA,self.BASEA_norm)
            #AB
            d2segment_AB = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.A,self.B,self.AB,self.AB_norm)
            #BC1
            d2segment_BC = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.B,self.C,self.BC,self.BC_norm)
            #C0C1
            d2segment_CD = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.C,self.D,self.CD,self.CD_norm)
            #C0D
            d2segment_DE = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.D,self.E,self.DE,self.DE_norm)
            #DE
            d2segment_EEND = self.distance_segment_to_segment(pre_Q0,pre_Q1,self.E,self.END,self.EEND,self.EEND_norm)                
            #BASE
            pre_O0O1=pre_Q1-pre_Q0
            d2segment_BASE = self.distance_point_to_segment(self.BASE,pre_Q0,pre_Q1,pre_O0O1,ca.norm_2(pre_O0O1))
            self.dis2obs[i+0] =   d2segment_BASEA
            self.dis2obs[i+1] =   d2segment_AB
            self.dis2obs[i+2] =   d2segment_BC
            self.dis2obs[i+3] =   d2segment_CD
            self.dis2obs[i+4] =   d2segment_DE
            self.dis2obs[i+5] =   d2segment_EEND
            self.dis2obs[i+6] =   d2segment_BASE
            safe_R=self.safe_statistic_R_list[k,:]
            h0 = self.dis2obs[i+0]- safe_R[0]
            h1 = self.dis2obs[i+1]- safe_R[1]
            h2 = self.dis2obs[i+2]- safe_R[2]
            h3 = self.dis2obs[i+3]- safe_R[3]
            h4 = self.dis2obs[i+4]- safe_R[4]
            h5 = self.dis2obs[i+5]- safe_R[5]
            h6 = self.dis2obs[i+6]- safe_R[6]
            for j in range(self.n_conpoment):
                # locals()[f'h{j}']=self.dis2obs[i+j]**2 - safe_R[j]**2 #作用域在这有点问题
                self.h[i+j]=locals()[f'h{j}']
                ###DERIVATIVE ALONG ANGLES
                locals()[f'dh{j}dx']=ca.jacobian(locals()[f'h{j}'],self.state)
                self.dhdx[i+j,:]=locals()[f'dh{j}dx']
                ### Lghx = G = dhdx @ gx
                locals()[f'G{j}']=locals()[f'dh{j}dx'] @ self.gx
                self.G[i+j,:]=locals()[f'G{j}']
                ### lg2hx = K = dhdx @ g2
                locals()[f'K{j}']=locals()[f'dh{j}dx'] @ self.g2
                self.K[i+j,:]=locals()[f'K{j}']
                if self.CBF_mode=='11':
                    # H0 = dh0dx@self.Ax + gama * h0 - self.param1[i+0] - self.param2[i+0] * (ca.norm_2(K0)**2) + dh0dp@Obs_v.T - ca.norm_2(dh0dp)*self.sigma_v
                    locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                    - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) 
                    
                if self.CBF_mode=='10':
                    # H0 = dh0dx@self.Ax + gama * h0 - self.param1[i+0] - self.param2[i+0] * (ca.norm_2(K0)**2) + dh0dp@Obs_v.T 
                    locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] \
                                    - self.param1[i+j] - self.param2[i+j] * (ca.norm_2(locals()[f'K{j}'])**2) 
                    
                if self.CBF_mode=='01':
                    # H0 = dh0dx@self.Ax + gama * h0 + dh0dp@Obs_v.T - ca.norm_2(dh0dp)*self.sigma_v
                    locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] 
                    
                if self.CBF_mode=='00':
                    # H0 = dh0dx@self.Ax + gama * h0 + dh0dp@Obs_v.T 
                    locals()[f'H{j}']=locals()[f'dh{j}dx']@self.Ax + gama * locals()[f'h{j}'] 

                self.H[i+j] = locals()[f'H{j}']

    def initial_CBF(self):
        for i in range(self.sphere_obstacle_num):
            Obstacle = self.sphere_obstacle_list[i,:]
            Obs_v = self.obs_v_list[i,:]
            safe_R = self.safe_R_list[i,:]
            self.initial_sphere_obstacle_CBF(Obstacle.T,safe_R,Obs_v,i)

        for i in range(self.capsule_obstacle_num):
            index=self.sphere_obstacle_num+i
            Obstacle = self.capsule_obstacle_list[i,:]
            Obs_v = self.obs_v_list[index,:]
            safe_R = self.safe_R_list[index,:]
            self.initial_capsule_obstacle_CBF(Obstacle.T,safe_R,Obs_v,i)

        for i in range(self.rectangle_obstacle_num):
            index=self.sphere_obstacle_num+self.sphere_obstacle_num+i
            Obstacle = self.rectangle_obstacle_list[i,:]
            Obs_v = self.obs_v_list[index,:]
            safe_R = self.safe_R_list[index,:]
            self.initial_rectangle_obstacle_CBF(Obstacle.T,safe_R,Obs_v,i)
        self.initial_base_CBF()
        self.initial_statistic_CBF()

    def caculate_dis2obs(self, states_input, 
                                ball_obstacle_input,
                                capsule_obstacle_input,
                                rectangle_obstacle_input,
                                Obs_v,
                                state_velocity):
        return self.F_dis2obs(states_input,ball_obstacle_input,capsule_obstacle_input,rectangle_obstacle_input,Obs_v,state_velocity)

    def caculate_barriers(self, states_input, 
                                ball_obstacle_input,
                                capsule_obstacle_input,
                                rectangle_obstacle_input,
                                safe_R_list,
                                Obs_v,
                                state_velocity):
        return self.F_barriers(states_input,safe_R_list,ball_obstacle_input,capsule_obstacle_input,rectangle_obstacle_input,Obs_v,state_velocity)

    def caculate_barriers_dx(self, states_input, 
                                ball_obstacle_input,
                                capsule_obstacle_input,
                                rectangle_obstacle_input,
                                safe_R_list,
                                Obs_v,
                                state_velocity):
        return self.F_dhdx(states_input,safe_R_list,ball_obstacle_input,capsule_obstacle_input,rectangle_obstacle_input,Obs_v,state_velocity)
    
    def caculate_F_H(self, states_input, 
                            ball_obstacle_input,
                            capsule_obstacle_input,
                            rectangle_obstacle_input,
                            safe_R_list,
                            Obs_v,
                            state_velocity):
        return self.F_H(states_input,safe_R_list,ball_obstacle_input,capsule_obstacle_input,rectangle_obstacle_input,Obs_v,state_velocity)

    def caculate_F_G(self, states_input, 
                            ball_obstacle_input,
                            capsule_obstacle_input,
                            rectangle_obstacle_input,
                            safe_R_list,
                            Obs_v,
                            state_velocity):
        return self.F_G(states_input,safe_R_list,ball_obstacle_input,capsule_obstacle_input,rectangle_obstacle_input,Obs_v,state_velocity)

    def caculate_F_K(self, states_input, 
                            ball_obstacle_input,
                            capsule_obstacle_input,
                            rectangle_obstacle_input,
                            safe_R_list,
                            Obs_v,
                            state_velocity):
        return self.F_K(states_input,safe_R_list,ball_obstacle_input,capsule_obstacle_input,rectangle_obstacle_input,Obs_v,state_velocity)

    """ def caculate_F_arm_point(self, states_input):
        return self.F_arm_point(states_input) """
    def solve_QP5(self,obstacles,
                        states_input, 
                        states_velocity=[0,0,0,0,0,0,0,0],
                        u_input=[0,0,0,0,0,0,0,0],
                        safe_R_list=None,
                        obs_v=None,
                        dt=None
                        ):
            ##return delta x and h_list
            #obstacless:ball_obstacle_input,capsule_obstacle_input,rectangle_obstacle_input

            if dt is not None and (self.CBF_mode == "10" or self.CBF_mode == "11"):
                self.dt =dt
            else:
                self.dt = np.zeros(self.u_len).reshape(self.u_len, 1)
            obs_v=obs_v
            k_d = 1.0
            h_val = self.caculate_barriers(states_input,obstacles[0],obstacles[1],obstacles[2],safe_R_list,obs_v,states_velocity)
            if hasattr(h_val, 'toarray'):
                h = h_val.toarray()
            else:
                h = np.array(h_val)
            h = np.ravel(h)
            h_val = self.caculate_barriers(states_input,obstacles[0],obstacles[1],obstacles[2],safe_R_list,obs_v,states_velocity)
            if hasattr(h_val, 'toarray'):
                h = h_val.toarray()
            else:
                h = np.array(h_val)
            h = np.ravel(h)
            h_min=min(h)
            self.is_slack=False 
            for i in range(len(h)):
                if h[i] <= self.h_threshold:
                    if h[i]>self.h_danger:
                        gama = (1-(self.h_threshold-h[i])/self.h_threshold)*self.normal_gama
                    elif h[i]<=self.h_danger and h[i] > 0:
                        gama = self.h_danger_p_gama
                    else:
                        gama = self.h_danger_n_gama
                    self.cal_H[i]   = self.F_H_list[i](states_input,safe_R_list,obstacles[0],obstacles[1],obstacles[2],gama,obs_v,states_velocity)
                    self.cal_G[i,:] = self.F_G_list[i](states_input,safe_R_list,obstacles[0],obstacles[1],obstacles[2],obs_v,states_velocity)
                    self.cal_K[i,:] = self.F_K_list[i](states_input,safe_R_list,obstacles[0],obstacles[1],obstacles[2],obs_v,states_velocity)
                else:
                    self.cal_H[i]   =ca.DM.zeros((1, 1))
                    self.cal_G[i,:] =ca.DM.zeros((1, self.u_len))
                    self.cal_K[i,:] =ca.DM.zeros((1, self.u_len))
            #set g_list for not_slack
            for i in range(self.totle_num):
                for j in range(self.n_conpoment):
                    index = i*self.n_conpoment+j
                    if h[index] <= self.h_threshold:
                         self.g_list[index] = self.cal_G[index,:]@self.u+k_d*self.cal_K[index,:]@self.dt+self.cal_H[index,:]
                    else:
                        self.g_list[index] = ca.DM(1)

            for i in range(self.n_base_cbf+self.n_statistic_cbf):
                index = (self.totle_num)*self.n_conpoment+i
                if h[index] <= self.h_threshold:
                        self.g_list[index] = self.cal_G[index,:]@self.u+k_d*self.cal_K[index,:]@self.dt+self.cal_H[index,:]
                else:
                        self.g_list[index] = ca.DM(1)
                
            index=self.n_CBF_constrain
            #state limite
            self.g_list[index: index+self.u_len] = -(self.u*self.T_step+states_input)+self.up_joint_limite
            self.g_list[index+self.u_len : index+2*self.u_len] = (self.u*self.T_step+states_input)-self.low_joint_limite
            #output limite
            self.g_list[index+2*self.u_len : index+3*self.u_len] = -(self.u - self.output_limite)
            self.g_list[index+3*self.u_len : index+4*self.u_len] = (self.u + self.output_limite)
            U_REF = ca.SX(u_input)
            cost = self.u.T @ self.H_mat @ self.u - 2 * U_REF.T @ self.H_mat @ self.u
            constraints = self.g_list[:index+4*self.u_len]
            qp = {'x':self.u, 'f':cost,  'g':constraints}
            qp_opts = {"print_time": 0, "printLevel": "none", "verbose": 0,"error_on_fail":False}
            
            # 第一次调用qpOASES求解器，使用SuppressOutput包装
            with SuppressOutput():
                try:
                    S = ca.qpsol('S', 'qpoases', qp,qp_opts)
                    r = S(lbg=self.lbg[:index+4*self.u_len])
                    u_opt = r['x']
                except Exception as e:
                    print(f"第一次QP求解失败: {e}")
                    u_opt = ca.DM.zeros(self.u_len)
                
            self.get_solution = False
            for i in range(self.u_len):
                if u_opt[i]!=0:
                    self.get_solution = True
                    self.u_last=u_opt
                    break
            if  self.get_solution == False:
                print(colored(("no solution!!"),color="red",attrs=["bold"]))
                # u_opt=self.u_last
                #set g_list for is_slack
                for i in range(self.totle_num):
                    for j in range(self.n_conpoment):
                        index = i*self.n_conpoment+j
                        if h[index] <= self.h_threshold:
                            self.g_list[index] = self.g_list[index]+self.slack[index]
                        else:
                            self.g_list[index] = ca.DM(1)

                for i in range(self.n_base_cbf+self.n_statistic_cbf):
                    index = (self.totle_num)*self.n_conpoment+i
                    if h[index] <= self.h_threshold:
                            self.g_list[index] = self.g_list[index]+self.slack[index]
                    else:
                            self.g_list[index] = ca.DM(1)
                
                U_REF = ca.vertcat(ca.SX(u_input),ca.SX.zeros(self.slack.size1()))
                cost_augmented = self.u_augmented.T @ self.H_mat_augmented @ self.u_augmented - 2 * U_REF.T @ self.H_mat_augmented @ self.u_augmented
                constraints_augmented = self.g_list
                qp = {'x':self.u_augmented, 'f':cost_augmented,  'g':constraints_augmented}
                qp_opts = {"print_time": 0, "printLevel": "none", "verbose": 0,"error_on_fail":False}
                
                # 第二次调用qpOASES求解器，使用SuppressOutput包装
                with SuppressOutput():
                    try:
                        S = ca.qpsol('S', 'qpoases', qp,qp_opts)
                        r = S(lbg=self.lbg)
                        u_opt = r['x'][:self.u.size1()]
                        optimal_slack = r['x'][self.u.size1():]
                    except Exception as e:
                        print(f"第二次QP求解失败: {e}")
                        u_opt = self.u_last if hasattr(self, 'u_last') else ca.DM.zeros(self.u_len)
                        optimal_slack = ca.DM.zeros(self.slack.size1())

            return u_opt,h
    








