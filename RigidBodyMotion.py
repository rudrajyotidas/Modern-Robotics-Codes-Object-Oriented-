import numpy as np

class RigidBodyMotion:

    def Rinv(self, rotMat):

        '''
        Input: A rotation matrix as a numpy arry
        Output: Its inverse

        The transpose of a rotation matrix is its inverse
        '''
        return rotMat.T

    def so3(self, vec):

        '''
        Input: A three-dimensional vector
        Output: A skew symmetric so3 matrix
        '''

        return np.array([[0, -vec[2], vec[1]],
                        [vec[2], 0, -vec[0]],
                        [-vec[1], vec[0], 0]])

    def so3Tovec(self, so3Mat):

        '''
        Input: A skew symmetric so3 matrix
        Output: A three-dimensional vector
        '''

        return np.array([so3Mat[2][1], so3Mat[0][2], so3Mat[1][0]])

    def AxisAndAngle(self, exp_coordinates):

        '''
        Input: Three dimensional exponential coordinates
        Output: Axis of rotation, and angle as a tuple
        '''

        n = np.linalg.norm(exp_coordinates)

        if n<1e-6:
            return np.zeros((1,3)), 0

        return (exp_coordinates/n, n)

    def so3MatExp(self, so3Mat):

        '''
        Input: so3 matrix
        Output: Matrix exponential of the so3 matrix
        '''

        axis, angle = self.AxisAndAngle(so3Mat)

        if abs(angle) < 1e-6:
            return np.eye(3)

        else:
            axisso3 = so3Mat/angle

            # Rodrigues' Formula
            return np.eye(3) + np.sin(angle)*axisso3 + (1-np.cos(angle))*(np.dot(axisso3, axisso3))

    def RotMatLog(self, R):

        '''
        Input: Rotation matrix R
        Output: so3 matrix, which is log of R
        '''

        acos = (np.trace(R) - 1)/2

        if acos >= 1:
            return np.zeros(3,3)

        elif acos <= -1:

            if abs(1 + R[2][2]) > 1e-6:
                omega = (1/np.sqrt(1 + R[2][2]))*(np.array([R[0][2], R[1][2], 1+R[2][2]]))

            elif abs(1 + R[1][1]) > 1e-6:
                omega = (1/np.sqrt(1 + R[1][1]))*(np.array([R[0][1], 1+R[1][1], R[2][1]]))

            else:
                omega = (1/np.sqrt(1 + R[0][0]))*(np.array([1+R[0][0], R[1][0], R[2][0]]))

            return self.so3(omega*np.pi)

        else:
            theta = np.arccos(acos)
            return (theta/(2*np.sin(theta)))(R - R.T)

    def Rp2Trans(self, R, p):

        '''
        Input: Rotation matrix R, and a translation vector p
        Output: Corresponding transformation matrix T
        '''

        return np.r_[np.c_[R, p], [np.array([0, 0, 0, 1])]]

    def Trans2Rp(self, T):

        '''
        Input: Transformation matrix T
        Output: R and p
        '''

        R = T[0:3, 0:3]
        p = T[0:3, 3]

        return R, p

    def Tinv(self, T):

        '''
        Input: Transformation matrix T
        Output: Inverse of the transformation matrix
        '''

        R, p = self.Trans2Rp(T)
        Rinv = self.Rinv(R)
        return np.r_[np.c_[Rinv, -np.dot(Rinv, p)], [[0,0,0,1]]]

    def Twist2se3(self, V):

        '''
        Input: Six-dimensional twist
        Output: se3 matrix'''

        omega = np.array([V[0], V[1], V[2]])
        v = np.array([V[3], V[4], V[5]])

        return np.r_[np.c_[self.so3(omega), v], np.zeros((1,4))]

    def se32Twist(self, se3):

        '''
        Input: se3 matrix
        Output: Six-dimensional twist
        '''

        return np.array([se3[2][1], se3[0][2], se3[1][0], se3[0][3], se3[1][3], se3[2][3]])

    def Ad(self, T):

        '''
        Input: Transformation matrix T
        Output: Adjoint transform of T
        '''

        R, p = self.Trans2Rp(T)

        return np.r_[np.c_[R, np.zeros(3,3)],
                    np.c_[np.dot(self.so3(p), R), R]]

    def ScrewAxisAndAngle(self, V):

        '''
        Input: Six dimensional twist
        Output: A screw axis and theta
        '''

        theta = np.linalg.norm([V[0], V[1], V[2]])

        if abs(theta) < 1e-6:
            theta = np.linalg.norm([V[3], V[4], V[5]])

        return V/theta, theta

    def se3MatExp(self, se3):

        '''
        Input: se3 matrix
        Output: SE3 matrix T
        '''

        omgtheta = self.so3Tovec(se3[0:3, 0:3])

        if np.linalg.norm(omgtheta) < 1e-6:
            return np.r_[np.c_[np.eye(3), se3[0:3,3]], [[0,0,0,1]]]

        else:
            theta = self.AxisAndAngle(omgtheta)[1]
            omegaso3 = se3[0:3, 0:3]/theta

            return np.r_[np.c_[self.so3MatExp(se3[0:3, 0:3]), np.dot(np.eye(3)*theta + (1-np.cos(theta))*omegaso3 + (theta - np.sin(theta))*np.dot(omegaso3, omegaso3), se3[0:3, 3])],
                            [[0,0,0,1]]]

    def Tlog(self, T):

        '''
        Input: SE3 matrix T
        Output: se3 representation of twist
        '''

        R, p = self.Trans2Rp(T)

        omgthetaso3 = self.RotMatLog(R)

        if np.array_equal(omgthetaso3, np.zeros(3,3)):
            return np.r_[np.c_[np.zeros(3,3), [T[0:3][3]]],
                            [[0,0,0,0]]]

        else:

            omega, theta = self.AxisAndAngle(omgthetaso3)

            Ginv = np.eye(3)*theta - omgthetaso3/2 + (1/theta + np.cot(theta/2)/2)*np.dot(omgthetaso3, omgthetaso3)

            return np.r_[np.c_[omgthetaso3, np.dot(Ginv, p)],
                            [[0,0,0,0]]]

    


