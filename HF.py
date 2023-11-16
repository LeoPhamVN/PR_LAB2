import numpy as np
from Histogram import *
import random

class HF:

    """
    Histogram Filter base class. Implements the histogram filter algorithm using a discrete Bayes Filter.
    """
    def __init__(self, p0, *args):
        """"
        The histogram filter is initialized with the initial probability histogram *p0* and the state transition probability matrix *Pk*. The state transition probability matrix is computed by the derived class through the pure virtual method *StateTransitionProbability*.
        The histogram filter is implemented as a discrete Bayes Filter. The state transition probability matrix is used in the prediction step and the measurement probability matrix is used in the update step.
        :param p0: initial probability histogram
        """
        self.num_bins_x = p0.num_bins_x
        self.num_bins_y = p0.num_bins_y
        self.nCells = self.num_bins_x * self.num_bins_y

        self.x_range = p0.x_range
        self.y_range = p0.y_range

        self.x_size = self.x_range[-1]
        self.y_size = self.y_range[-1]
        self.cell_size_x = 2 * self.x_size / self.num_bins_x
        self.cell_size_y = 2 * self.y_size / self.num_bins_y

        self.p0 = Histogram2D(self.num_bins_x, self.num_bins_y, self.x_range, self.y_range)
        self.p0.histogram_1d= p0.histogram_1d.copy()
        self.pk_1 = Histogram2D(self.num_bins_x, self.num_bins_y, self.x_range, self.y_range)
        self.pk_1.histogram_1d= self.p0.histogram_1d.copy()
        self.pk_hat = Histogram2D(self.num_bins_x, self.num_bins_y, self.x_range, self.y_range)
        self.pk = Histogram2D(self.num_bins_x, self.num_bins_y, self.x_range, self.y_range)

        Q = 0.5
        Re = 1.0
        self.Q  = Q         # Initialize the motion model noise
        self.Re = Re        # Initialize the measurement noise

        # Compute State Transition Probability matrixs, save in "StateTransitionProbability.npy" file, load it at the begginning of the simulation
        try:
            self.Pk=np.load("StateTransitionProbability.npy", allow_pickle=True)
        except:
            Pk_tuple = self.StateTransitionProbability()
            np.save("StateTransitionProbability", Pk_tuple)
            self.Pk=np.load("StateTransitionProbability.npy", allow_pickle=True)

        # Compute Measurement Probability matrixs, save in "NormalProbability.npy" file, load it at the begginning of the simulation
        try:
            self.Nk=np.load("NormalProbability.npy", allow_pickle=True)
        except:
            Nk_tuple = self.NormalProbability(Re)      # std: 1.0 : standard deviation of the measurement
            np.save("NormalProbability", Nk_tuple)
            self.Nk=np.load("NormalProbability.npy", allow_pickle=True)

        super().__init__(*args)

    def ToCell(self, displacemt):
        """
        Converts a metric displacement to a cell displacement.

        :param displacemt: input displacement in meters
        :return: displacement in cells
        """
        cell=int( displacemt / self.cell_size_x)

        return cell

    def DiscretizeInput(self, uk):
        """
        Discretizes the control input *u*. To be overriden by the derived class.
        :param u: control input
        :return: discretized control input
        """

        pass

    def StateTransitionProbability(self):
        """
        Returns the state transition probability matrix.
        This is a pure virtual method that must be implemented by the derived class.

        :return: *Pk* state transition probability matrix
        """
        pass

    def StateTransitionProbability_4_xk_1_uk(self, pk_1, cell_uk):
        """
        Returns the state transition probability matrix for the given control input *uk* and *pk_1*.
        This is a pure virtual method that must be implemented by the derived class.

        :param etak_1: previous robot pose in cells
        :param uk: input displacement in number of cells
        :return: state transition probability :math:`p_k=p(\eta_k | \eta_{k-1}, u_k)`
        """
        pass

    def NormalProbability(self, std):
        """
        Returns the normal distribution with zero mean for the given distance input *uk*, standard deviation *std*.
        This is a pure virtual method that must be implemented by the derived class.

        :return: Normal zero mean probability :math:`n_k~N(0, std)`
        """
        pass
        
    def MeasurementProbability(self,zk):
        """
        Returns the measurement probability matrix for the given measurement *zk*.
        This is a pure virtual method that must be implemented by the derived class.

        :param zk: measurement.
        :return: *pzk* measurement probability histogram
        """

        pass

    def uk2cell(self, uk):
        """"
        Converts the number of cells the robot has displaced along its DOFs in the world N-Frame to an index that can be
        used to acces the state transition probability matrix.
        This is a pure virtual method that must be implemented by the derived class.

        :param uk: vector containing the number of cells the robot has displaced in all the axis of the world N-Frame
        :returns: index: index that can be used to access the state transition probability matrix
        """
        pass

    def Prediction(self, pk_1, uk):
        """
        Computes the prediction step of the histogram filter. Given the previous probability histogram *pk_1* and the
        control input *uk*, it computes the predicted probability histogram *pk_hat* after the robot displacement *uk*
        according to the motion model described by the state transition probability.

        :param pk_1: previous probability histogram
        :param uk: control input
        :return: *pk_hat* predicted probability histogram
        """

        # cell_uk= self.uk2cell(uk)     
        self.StateTransitionProbability_4_xk_1_uk(pk_1, uk)

        return self.pk_hat

    def Update(self,pk_hat, zk):
        """
        Computes the update step of the histogram filter. Given the predicted probability histogram *pk_hat* and the measurement *zk*, it computes first the measurement probability histogram *pzk* and then uses the Bayes Rule to compute the updated probability histogram *pk*.
        :param pk_hat: predicted probability histogram
        :param zk: measurement
        :return: pk: updated probability histogram
        """
        self.pk = self.MeasurementProbability(zk) * pk_hat
        self.pk.histogram_1d /= np.sum(self.pk.histogram_1d)

        return self.pk

