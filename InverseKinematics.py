import numpy as np

from ForwardKinematics import ForwardKinematics
from Jacobian import Jacobian

class InverseKinematics(ForwardKinematics, Jacobian):

    def __init__(self, max_iter):
        self.max_iter = 20

    def IKbody(self, SbList, M, T, thetaList0, err_omega, err_v):

        '''
        SbList: Screw axes of each joint in the end effector frame
                when the manipulator is in home position. Data is passed
                in the form of a matrix with the screw axes as the columns
                
        thetaList: List of joint angles and displacements

        M: T of end effector frame t home position
        
        T: Target frame of end effector
        
        thetaList0: Guess of joint angles
        
        err_omega: Error tolerance for orientation
        
        err_v: Error tolerance for linear position

        Output: List of joint angles, and success value
        '''

        thetaList = thetaList0.copy()
        i=0

        Vb = self.se32Twist(self.Tlog(np.dot(self.Tinv(self.FKBody(M, SbList, thetaList))), T))

        err_not_satisfied = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > err_omega or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > err_v

        while err_not_satisfied and i < self.max_iter:

            thetaList = thetaList + np.dot(np.linalg.pinv(self.BodyJacobian(SbList, thetaList)), Vb)
            Vb = self.se32Twist(self.Tlog(np.dot(self.Tinv(self.FKBody(M, SbList, thetaList))), T))
            i = i+1
            err_not_satisfied = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > err_omega or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > err_v

        return thetaList, not err_not_satisfied


    def IKspace(self, SsList, M, T, thetaList0, err_omega, err_v):

        '''
        SbList: Screw axes of each joint in the end effector frame
                when the manipulator is in home position. Data is passed
                in the form of a matrix with the screw axes as the columns
                
        thetaList: List of joint angles and displacements

        M: T of end effector frame t home position
        
        T: Target frame of end effector
        
        thetaList0: Guess of joint angles
        
        err_omega: Error tolerance for orientation
        
        err_v: Error tolerance for linear position

        Output: List of joint angles, and success value
        '''

        thetaList = thetaList0.copy()
        i=0

        Tsb = self.FKSpace(M, SsList, thetaList)
        Vb = self.se32Twist(self.Tlog(np.dot(self.Tinv(Tsb), T)))
        Vs = np.dot(self.Ad(Tsb), Vb)

        err_not_satisfied = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > err_omega or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > err_v

        while err_not_satisfied and i < self.max_iter:

            thetaList = thetaList + np.dot(np.linalg.pinv(self.SpaceJacobian(SsList, thetaList)), Vs)
            Tsb = self.FKSpace(M, SsList, thetaList)
            Vb = self.se32Twist(self.Tlog(np.dot(self.Tinv(Tsb), T)))
            Vs = np.dot(self.Ad(Tsb), Vb)
            i = i+1
            err_not_satisfied = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > err_omega or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > err_v

        return thetaList, not err_not_satisfied




