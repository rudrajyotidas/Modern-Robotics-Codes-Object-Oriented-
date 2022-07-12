import numpy as np

from RigidBodyMotion import RigidBodyMotion

class OpenChainDynamics(RigidBodyMotion):

    def ad(self, V):

        '''
        Input: Twist
        
        Output: 6x6 [adV] matrix
        '''

        omgso3 = self.so3(np.array([V[0], V[1], V[2]]))
        vso3 = self.so3(np.array([V[3],V[4],V[5]]))

        return np.r_[np.c_[omgso3, np.zeros((3,3))],
                    np.c_[vso3, omgso3]]

    def NewtonEulerInverseDynamics(self, thetaList, dthetaList, ddthetaList, g, Ftip, Mlist, Glist, Sslist):

        '''
        Newton Euler Inverse Dynamics for Open Chains
        
        thetaList: Joint angles
        dthetaList: Joint angular velocity
        ddthetaList: Joint angular acceleration
        
        g: Gravity vector
        
        Ftip: End effector wrench to be generated
        
        Mlist: List of link frames {i} in {i-1} frame, [M01, M12,....]
        
        Glist: Spatial inertia matrix of the links
        
        Sslist: Screw axes of joints in space frame
        
        Output: Joint forces or torques
        '''

        n = len(thetaList)
        T = np.eye(4)

        # Ai is the screw axis of joint i in {i}
        Ai = np.zeros((6,n))

        AdTi = [[None]] * (n + 1)

        Vi = np.zeros((6, n+1))
        Vdi = np.zeros((6, n+1))

        Vdi[:, 0] = np.r_[[0,0,0], -g]

        AdTi[n] = self.Ad(self.Tinv(Mlist[n]))

        Fi = Ftip.copy()

        tau = np.zeros(n)

        #------- Forward Iteration ----------
        for i in range(n):

            T = np.dot(T, Mlist[i])
            Ai[:, i] = np.dot(self.Ad(T), Sslist[:, i])
            Ti = np.dot(self.se3MatExp(-self.Twist2se3(Ai[:,i])*thetaList[i]), self.Tinv(Mlist[i]))
            AdTi[i] = self.Ad(Ti)
            Vi[:, i+1] = np.dot(AdTi[i], Vi[:, i]) + np.dot(Ai[:,i], dthetaList[i])
            Vdi[:, i+1] = np.dot(AdTi[i], Vdi[:,i]) + np.dot(self.ad(Vi[:,i+1]), Ai[:,i])*dthetaList[i] + Ai[:,i]*ddthetaList[i]

        #------- Backward Iteration ---------
        for i in range(n-1, -1, -1):
            Fi = np.dot(AdTi[i+1].T, Fi) + np.dot(Glist[i], Vdi[:,i+1]) - np.dot(self.ad(Vi[:,i+1]).T, np.dot(Glist[i], Vi[:, i+1]))
            tau[i] = np.dot(Fi.T, Ai[:,i])

        return tau

    def MassMatrix(self, thetaList, Mlist, Glist, Sslist):

        '''
        Find mass matrix of manipulator at a particular configurations given spatial inertia matrices of each link

        thetaList: Joint angles and displacements
        
        Mlist: List of link frames {i} in {i-1} frame, [M01, M12,....]
        
        Glist: Spatial inertia matrix of the links
        
        Sslist: Screw axes of joints in space frame
        '''

        n = len(thetaList)
        M = np.zeros(n,n)

        for i in range(n):
            dthetaList = np.array([0]*n)
            ddthetaList = np.array([0]*n)

            ddthetaList[i] = 1

            M[:, i] = self.NewtonEulerInverseDynamics(thetaList, dthetaList, ddthetaList, np.array([0,0,0]),
                                np.array([0,0,0,0,0,0]), Mlist, Glist, Sslist)

        return M

    def VelQuadTerms(self, thetaList, dthetaList, Mlist, Glist, Sslist):

        '''
        Coriolis terms in the Lagrange equation of motion

        thetaList: Joint angles and displacements

        dthetaList: Joint angular velocities and linear velocities
        
        Mlist: List of link frames {i} in {i-1} frame, [M01, M12,....]
        
        Glist: Spatial inertia matrix of the links
        
        Sslist: Screw axes of joints in space frame
        '''

        return self.NewtonEulerInverseDynamics(thetaList, dthetaList, np.array([0]*len(thetaList)), 
                                                    np.array([0,0,0]), np.array([0,0,0,0,0,0]), 
                                                    Mlist, Glist, Sslist)

    def GravityTerms(self, thetaList, g, Mlist, Glist, Sslist):

        '''
        Gravity terms in the Lagrange equation of motion

        thetaList: Joint angles and displacements

        g: Gravity vector
        
        Mlist: List of link frames {i} in {i-1} frame, [M01, M12,....]
        
        Glist: Spatial inertia matrix of the links
        
        Sslist: Screw axes of joints in space frame
        '''

        return self.NewtonEulerInverseDynamics(thetaList, np.array([0]*len(thetaList)), np.array([0]*len(thetaList)), 
                                                    g, np.array([0,0,0,0,0,0]), 
                                                    Mlist, Glist, Sslist)

    def EndEffectorTerms(self, thetaList, Ftip, Mlist, Glist, Sslist):

        '''
        End effector force terms in the Lagrange equation of motion

        thetaList: Joint angles and displacements

        Ftip: Wrench applied by end effector
        
        Mlist: List of link frames {i} in {i-1} frame, [M01, M12,....]
        
        Glist: Spatial inertia matrix of the links
        
        Sslist: Screw axes of joints in space frame
        '''

        return self.NewtonEulerInverseDynamics(thetaList, np.array([0]*len(thetaList)), np.array([0]*len(thetaList)), 
                                                    np.array([0,0,0]), Ftip, 
                                                    Mlist, Glist, Sslist)

    def ForwardDynaamics(self, thetaList, dthetaList, tau, g, Ftip, Mlist, Glist, Sslist):

        '''
        Forward Dynamics for open chain

        thetaList: Joint angles and displacements

        dthetaList: Joint angular velocities and linear velocities

        tau: Joint torques and forces

        g: Gravity vector

        Ftip: Wrench applied by end effector
        
        Mlist: List of link frames {i} in {i-1} frame, [M01, M12,....]
        
        Glist: Spatial inertia matrix of the links
        
        Sslist: Screw axes of joints in space frame
        '''

        return np.dot(np.linalg.inv(self.MassMatrix(thetaList, Mlist, Glist, Sslist)), 
                        (tau - self.VelQuadTerms(thetaList, dthetaList, Mlist, Glist, Sslist) \
                            - self.GravityTerms(thetaList, g, Mlist, Glist, Sslist)\
                                - self.EndEffectorTerms(thetaList, Ftip, Mlist, Glist, Sslist)))

    def EulerStep(self, thetaList, dthetaList, ddthetaList, dt):

        '''
        Calculates state of open chain at next time step
        '''

        return thetaList + dt*dthetaList, dthetaList + dt*ddthetaList



