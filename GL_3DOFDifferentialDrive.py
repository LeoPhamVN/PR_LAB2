from GL import GL
from DifferentialDriveSimulatedRobot import *
from DR_3DOFDifferentialDrive import *
from scipy import stats
from Pose3D import *
from Histogram import *

class GL_3DOFDifferentialDrive(GL, DR_3DOFDifferentialDrive):
    """
    Grid Reckoning Localization for a 3 DOF Differential Drive Mobile Robot.
    """

    def __init__(self, dx_max, dy_max, range_dx, range_dy, p0, index, kSteps, robot, x0, *args):
        """
        Constructor of the :class:`GL_4DOFAUV` class. Initializes the Dead reckoning localization algorithm as well as the histogram filter algorithm.

        :param dx_max: maximum x displacement in meters
        :param dy_max: maximum y displacement in meters
        :param range_dx: range of x displacements in meters
        :param range_dy: range of y displacements in meters
        :param p0: initial probability histogram
        :param index: index struture containing plotting information
        :param kSteps: number of time steps to simulate the robot motion
        :param robot: robot object
        :param x0: initial robot pose
        :param args: additional arguments
        """

        super().__init__(p0, index, kSteps, robot, x0, *args)

        self.sigma_d = 1

        self.range_dx = range_dx
        self.range_dy = range_dy

        self.Deltax = 0  # x displacement since last x cell motion
        self.Deltay = 0  # y displacement since last y cell motion
        self.Delta_etak = Pose3D(np.zeros((3, 1)))

        self.cell_size = self.pk_1.cell_size_x  # cell size is the same for x and y

        self.travelled_distance = np.array([[0.],[0.]])    # Travelled distance of the robot in the xOy (N-Frame) [met]
        self.xsk_1_cell = np.array([[0],[0]])

    def GetMeasurements(self):
        """
        Read the measurements from the robot. Returns a vector of range distances to the map features.
        Only those features that are within the :attr:`SimulatedRobot.SimulatedRobot.Distance_max_range` of the sensor are returned.
        The measurements arribe at a frequency defined in the :attr:`SimulatedRobot.SimulatedRobot.Distance_feature_reading_frequency` attribute.

        :return: vector of distances to the map features
        """
        # TODO: To be implemented by the student
        ranges, R_ranges  = self.robot.ReadRanges()
        return ranges, R_ranges

    def StateTransitionProbability_4_uk(self,uk):
        return self.Pk[uk]
    
    def StateTransitionProbability_4_xk_1_uk(self, pk_1, cell_uk):
        """
        Computes the state transition probability histogram given the previous robot pose :math:`\eta_{k-1}` and the input :math:`u_k`:

        .. math::

            p(\eta_k | \eta_{k-1}, u_k)

        :param etak_1: previous robot pose in cells
        :param uk: input displacement in number of cells
        :return: state transition probability :math:`p_k=p(\eta_k | \eta_{k-1}, u_k)`

        """
        # TODO: To be implemented by the student
        # Motion UP and DOWN state transition probability
        self.pk_hat.histogram_2d = self.StateTransitionProbability_4_uk(int(cell_uk[1])) @ pk_1.histogram_2d
        # Motion LEFT and RIGHT state transition probability
        self.pk_hat.histogram_2d = (self.StateTransitionProbability_4_uk(int(cell_uk[0])) @ self.pk_hat.histogram_2d.T).T
        
        return True

    def StateTransitionProbability(self):
        """
        Computes the complete state transition probability matrix. The matrix is a :math:`n_u \times m_u \times n^2` matrix,
        where :math:`n_u` and :math:`m_u` are the number of possible displacements in the x and y axis, respectively, and
        :math:`n` is the number of cells in the map. For each possible displacement :math:`u_k`, each previous robot pose
        :math:`{x_{k-1}}` and each current robot pose :math:`{x_k}`, the probability :math:`p(x_k|x_{k-1},u_k)` is computed.


        :return: state transition probability matrix :math:`P_k=p{x_k|x_{k-1},uk}`
        """

        # TODO: To be implemented by the student
        # Standard deviation of the transition noise [cell]
        std = self.Q
        N_tuple = self.NormalProbability(std)

        # Find max between num_bins_y and num_bins_x
        num_bins_max = max(self.num_bins_x, self.num_bins_y)
        # Initalise P_tuple
        P_tuple = ()
        for uk in range(num_bins_max):
            # Initalize Transition Probability matrix P
            P = np.zeros((self.num_bins_x,self.num_bins_y))
            for index_cols in range(self.num_bins_y):
                for index_rows in range(self.num_bins_x // 2):
                    # Compute index
                    index_pos = (index_cols + uk - index_rows) % self.num_bins_x
                    index_neg = (index_cols + uk + index_rows) % self.num_bins_y
                    # Set value of the P matrix
                    P[index_pos][index_cols] = N_tuple[index_rows]
                    P[index_neg][index_cols] = P[index_pos][index_cols]
            P_tuple = P_tuple + (P,)
        return P_tuple

    def uk2cell(self, uk):
        """
        Converts the number of cells the robot has displaced along its DOFs in the world N-Frame to an index that can be
        used to acces the state transition probability matrix.

        :param uk: vector containing the number of cells the robot has displaced in all the axis of the world N-Frame
        :returns: index: index that can be used to access the state transition probability matrix
        """

        # TODO: To be implemented by the student
        # I do not use this function
        pass

    def NormalProbability(self, std):
        """
        Returns the normal distribution with zero mean for the given distance input *uk*, standard deviation *std*.
        This is a pure virtual method that must be implemented by the derived class.

        :return: Normal zero mean probability :math:`n_k~N(0, std)`
        """

        # Find max between num_bins_y and num_bins_x
        num_bins_max = max(self.num_bins_x, self.num_bins_y)

        if std == 0.0:
            # If std = 0, it is not the normal distribution, only one element is 1, the remainning are 0
            N_tuple = ()
            N_tuple = N_tuple + (1,)
            for uk in range(num_bins_max+1):
                N_tuple = N_tuple + (0,)
        else:
            # If std >< 0, it is the normal distribution
            N_tuple = ()
            for uk in range(num_bins_max+2):
                n = 1 / np.sqrt(2*np.pi*std**2) * np.exp(-(uk)**2 / (2 * std**2))
                N_tuple = N_tuple + (round(n, 2),)    

        return N_tuple

    def MeasurementProbability(self, zk):
        """
        Computes the measurement probability histogram given the robot pose :math:`\eta_k` and the measurement :math:`z_k`.
        In this case the measurement is the vector of the distances to the landmarks in the map.

        :param zk: :math:`z_k=[r_0~r_1~..r_k]` where :math:`r_i` is the distance to the i-th landmark in the map.
        :returns: Measurement probability histogram :math:`p_z=p(z_k | \eta_k)`
        """

        # TODO: To be implemented by the student
        # Initialise measurement probability
        p_z = Histogram2D(self.num_bins_x, self.num_bins_y, self.x_range, self.y_range)

        for x in p_z.x_range:
            for y in p_z.y_range:
                # Calculate the distance from the cell need to be computed the probability to the landmarks
                dst1 = (abs(np.linalg.norm(np.array([[x],[y]]) - self.robot.M[0]) - zk[0])).astype(int)
                dst2 = (abs(np.linalg.norm(np.array([[x],[y]]) - self.robot.M[1]) - zk[1])).astype(int)
                dst3 = (abs(np.linalg.norm(np.array([[x],[y]]) - self.robot.M[2]) - zk[2])).astype(int)

                # Compute the probability with each landmark
                p1 = self.Nk[dst1]
                p2 = self.Nk[dst2]
                p3 = self.Nk[dst3]

                # Compute the probability with 3 landmarks
                p_z.element[x,y] = p1 * p2 * p3
        return p_z

    def GetInput(self,usk):
        """
        Provides an implementation for the virtual method :meth:`GL.GetInput`.
        Gets the number of cells the robot has displaced in the x and y directions in the world N-Frame. To do it, it
        calls several times the parent method :meth:`super().GetInput`, corresponding to the Dead Reckoning Localization
        of the robot, until it has displaced at least one cell in any direction.
        Note that an iteration of the robot simulation :meth:`SimulatedRobot.fs` is normally done in the :meth:`GL.LocalizationLoop`
        method of the :class:`GL.Localization` class, but in this case it is done here to simulate the robot motion
        between the consecutive calls to :meth:`super().GetInput`.

        :param usk: control input of the robot simulation
        :return: uk: vector containing the number of cells the robot has displaced in the x and y directions in the world N-Frame
        """
        # TODO: To be implemented by the student
        # Here, I computed the displacements following 2 methods: from velocity and from the position
        # The first method, the accuracy is low because the noise in the robot simulation function
        # The second method, the accuracy is higher
        # Compute the displacements of the robot from velocity
        linear_velocity     = float(usk[0])
        angular_velocity    = float(usk[1])
        heading_robot       = float(self.robot.xsk[2])

        self.travelled_distance += np.array([[linear_velocity * self.robot.dt * cos(heading_robot + angular_velocity * self.robot.dt)],
                                             [linear_velocity * self.robot.dt * sin(heading_robot + angular_velocity * self.robot.dt)]])
        
        # Compute the displacements of the robot from the position of robot
        self.travelled_distance = np.array([(self.robot.xsk[0]).astype(float), (self.robot.xsk[1]).astype(float)])

        # Calculate displacement in cell
        displacement_cell = (np.sign(self.travelled_distance) * (abs(self.travelled_distance) // self.cell_size) - self.xsk_1_cell).astype(int)

        # Calculate position of the robot in cell at this step, which will be used in the next step to compute displacement in cell
        self.xsk_1_cell = np.sign(self.travelled_distance) * (abs(self.travelled_distance) // self.cell_size)

        return displacement_cell

        



