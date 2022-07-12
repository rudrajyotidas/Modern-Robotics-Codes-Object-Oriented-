import numpy as np

from RigidBodyMotion import RigidBodyMotion

class ForwardKinematics(RigidBodyMotion):

    def FKBody(self, M, SbList, thetaList):

        '''
        Forward Kinematics in body frame

        SbList: Screw axes of each joint in the end effector frame
                when the manipulator is in home position. Data is passed
                in the form of a matrix with the screw axes as the columns
                
        thetaList: List of joint angles and displacements

        M: T of end effector frame t home position

        Output: T to represent end effector frame
        '''

        T = M.copy()

        for i in range(len(thetaList)):
            T = np.dot(T, self.se3MatExp(self.Twist2se3(SbList[:, i])*thetaList[i]))

        return T

    def FKSpace(self, M, SsList, thetaList):

        '''
        Forward Kinematics in space frame

        SsList: Screw axes of each joint in the end space frame
                when the manipulator is in home position. Data is passed
                in the form of a matrix with the screw axes as the columns
                
        thetaList: List of joint angles and displacements

        M: T of end effector frame t home position

        Output: T to represent end effector frame
        '''

        T = M.copy()

        for i in range(len(thetaList)-1, -1, -1):
            T = np.dot(self.se3MatExp(self.Twist2se3(SsList[:, i])*thetaList[i], T))

        return T

