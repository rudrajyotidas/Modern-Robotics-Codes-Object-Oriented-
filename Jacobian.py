import numpy as np

from RigidBodyMotion import RigidBodyMotion

class Jacobian(RigidBodyMotion):

    def BodyJacobian(self, SbList, thetaList):

        '''
        Output: Body Jacobian at current configuration

        SbList: Screw axes of each joint in the end effector frame
                when the manipulator is in home position. Data is passed
                in the form of a matrix with the screw axes as the columns
                
        thetaList: List of joint angles and displacements

        M: T of end effector frame t home position
        '''

        Jb = SbList.copy().astype(np.float)

        T = np.eye(4)

        for i in range(len(thetaList)-2, -1, -1):

            T = np.dot(T, self.se3MatExp(-self.Twist2se3(SbList[:, i+1])*thetaList[i+1]))

            Jb[:, i] = np.dot(self.Ad(T), SbList[:,i])

        return Jb

    def SpaceJacobian(self, SsList, thetaList):

        '''
        Output: Body Jacobian at current configuration

        SsList: Screw axes of each joint in the space frame
                when the manipulator is in home position. Data is passed
                in the form of a matrix with the screw axes as the columns
                
        thetaList: List of joint angles and displacements

        M: T of end effector frame t home position
        '''

        Js = SsList.copy().astype(np.float)

        T = np.eye(4)

        for i in range(1, len(thetaList)):

            T = np.dot(T, self.se3MatExp(self.Twist2se3(SsList[:, i-1])*thetaList[i-1]))

            Js[:, i] = np.dot(self.Ad(T), SsList[:,i])

        return Js