from enum import Enum
import numpy as np
from matlab_worker import MatlabWorker
from abc import ABC, abstractmethod
import scipy.io as sio


class DREXDistribution(Enum):
    """Distribution types supported by D-REX"""
    GAUSSIAN = "gaussian"
    LOGNORMAL = "lognormal"
    GMM = "gmm"
    POISSON = "poisson"


class DREXPrior(ABC):
    """Abstract class representing one prior distribution"""
    def __init__(self, drexDistribution: DREXDistribution):
        if isinstance(drexDistribution, DREXDistribution):
            self.drexDistribution = drexDistribution
        else:
            raise ValueError("drexDistribution invalid! Value must be member of enum DREXDistribution.")

    @abstractmethod
    def to_dict(self):
        """Returns a dictionary representing the prior"""
        pass


class DREXGaussianPrior(DREXPrior):
    """
    Gaussian prior with specific values of mu, ss, n.

    After using D-REX's estimate_suffstat.m function, a 1x1 struct with fields mu, ss, n is returned. Each field contains a 1x1 cell (one for each feature), in which there is a single value containing a column vector with +D+ values.
    """

    def __init__(self, mu, ss, n):
        super().__init__(DREXDistribution.GAUSSIAN)
        self.mu = mu
        self.ss = ss
        self.n = n

    def to_dict(self):
        return {'mu': np.array([self.mu], dtype=object), 'ss': np.array([self.ss], dtype=object),
                'n': np.array([self.n], dtype=object)}


class DREXLognormalPrior(DREXPrior):
    """Log-scaled Gaussian (=normal) prior with specific values of mu, ss, n."""

    def __init__(self, mu, ss, n):
        super().__init__(DREXDistribution.LOGNORMAL)
        self.mu = mu
        self.ss = ss
        self.n = n

    def to_dict(self):
        return {'mu': np.array([self.mu], dtype=object), 'ss': np.array([self.ss], dtype=object),
                'n': np.array([self.n], dtype=object)}


class DREXGmmPrior(DREXPrior):
    """Gaussin Mixture Model (GMM) prior"""

    def __init__(self, mu, sigma, n, pi, sp, k):
        super().__init__(DREXDistribution.GMM)
        self.mu = mu
        self.sigma = sigma
        self.n = n
        self.pi = pi
        self.sp = sp
        self.k = k

    def to_dict(self):
        return {'mu': self.mu, 'sigma': self.sigma, 'n': self.n,
                'pi': self.pi, 'sp': self.sp, 'k': np.array([self.k])}


class DREXPoissonPrior(DREXPrior):
    """Poisson prior with specific values of lambda and n."""

    def __init__(self, lambd, n):
        super().__init__(DREXDistribution.POISSON)
        self.lambd = lambd
        self.n = n

    def to_dict(self):
        return {'lambda': np.array([self.lambd], dtype=object), 'n': np.array([self.n], dtype=object)}


class DREXPriorInputParameters:
    """This class contains instructions for D-REX's estimate_suffstat.m function, which outputs a file that is parsed
    by DREXPriorOutputParameters. """

    def __init__(self, distribution: DREXDistribution, sequence, DOrMaxNComp=None):
        self._distribution = distribution
        self._DOrMaxNComp = DOrMaxNComp
        self._sequence = np.array([sequence], dtype='d').T  # (dim: time x trial x feature)

    def with_output_file_path(self, output_file_path):
        self._output_file_path = output_file_path
        return self

    def write_mat(self, filename):
        DOrMaxNCompKey = "max_ncomp" if self._distribution == DREXDistribution.GMM else "D"
        mat_data = {
            "xs": self._sequence,
            "params": {
                "distribution": self._distribution.value,
                DOrMaxNCompKey: self._DOrMaxNComp
            },
            "output_file_path": self._output_file_path
        }
        if self._DOrMaxNComp == None:
            del mat_data["params"][DOrMaxNCompKey]

        sio.savemat(filename, mat_data)

        return filename


class DREXPriorOutputParameters:
    """This class parses the output of D-REX's estimate_suffstat.m function and returns a DREXDistribution."""

    def __init__(self):
        pass

    def from_mat(file_path):
        mat_data = sio.loadmat(file_path)
        data_set = mat_data["drex_out"]
        distribution = DREXDistribution(mat_data["distribution"])
        columns = data_set.dtype.names
        data = {c: data_set[c][0, 0][0][0] for c in columns}

        if distribution == DREXDistribution.GAUSSIAN:  # ToDo: support multiple "feature dimensions"
            result = DREXGaussianPrior(data['mu'], data['ss'], data['n'])
        elif distribution == DREXDistribution.LOGNORMAL:
            result = DREXLognormalPrior(data['mu'], data['ss'], data['n'])
        elif distribution == DREXDistribution.GMM:
            result = DREXGmmPrior(data['mu'], data['sigma'], data['n'], data['pi'], data['sp'], data['k'])
        elif distribution == DREXDistribution.POISSON:
            result = DREXPoissonPrior(data['lambda'], data['n'])

        return result


class DREXPriorEstimator:
    """This class triggers the D-REX's prior calculation."""

    def __init__(self, model_io_paths):
        self._matlab_worker = MatlabWorker()
        self._model_io_paths = model_io_paths

    def estimate(self, distribution, sequence, DOrMaxNComp=None):
        """Given the desired distribution and the input sequence, D-REX's estimate_suffstat.m function is triggered."""
        input_file_path = self._model_io_paths["generic_filename_prefix"] + "estimate-prior_input.mat"
        output_file_path = self._model_io_paths["generic_filename_prefix"] + "estimate-prior_output.mat"
        DREXPriorInputParameters(distribution, DOrMaxNComp, sequence).with_output_file_path(output_file_path).write_mat(
            input_file_path)
        return self._matlab_worker.estimate_prior(input_file_path)


class DREXInputParameters:
    """This class represents all of D-REX's input parameters."""

    def __init__(self, distribution, D, prior, hazard, obsnz, memory, maxhyp):
        self._distribution = distribution
        self._D = D
        self._prior = prior
        self._hazard = hazard
        self._obsnz = obsnz
        self._memory = memory
        self._maxhyp = maxhyp
        self._sequence = None

    def with_sequence(self, sequence):
        self._sequence = np.array([sequence], dtype='d').T
        return self

    def with_output_file_path(self, output_file_path):
        self._output_file_path = output_file_path
        return self

    def write_mat(self, filename):
        DOrMaxNCompKey = "max_ncomp" if self._distribution == DREXDistribution.GMM.value else "D"
        mat_data = {
            "x": self._sequence,
            "output_file_path": self._output_file_path,
            "params": {
                "distribution": self._distribution,
                DOrMaxNCompKey: self._D,
                "prior": self._prior,
                "hazard": self._hazard,
                "obsnz": self._obsnz,
                "memory": self._memory,
                "maxhyp": self._maxhyp,
            }
        }

        sio.savemat(filename, mat_data)

        return filename

    def read_csv(filename):
        return DREXInputParameters()


class DREXOutputParameters:
    """This class represents all of D-REX's output parameters."""

    def __init__(self, source_file_path, input_file_path, input_sequence, psi, distribution, surprisal, joint_surprisal,
                 context_beliefs, change_decision_threshold, change_decision_changepoint, change_decision_probability,
                 belief_dynamics):
        self.source_file_path = source_file_path
        self.input_file_path = input_file_path
        self.input_sequence = input_sequence
        self.psi = psi
        self.distribution = distribution
        self.surprisal = surprisal
        self.joint_surprisal = joint_surprisal
        self.context_beliefs = context_beliefs
        self.change_decision_threshold = change_decision_threshold
        self.change_decision_changepoint = change_decision_changepoint
        self.change_decision_probability = change_decision_probability
        self.belief_dynamics = belief_dynamics

    def entropy_of(self, ensemble):
        entropy = 0
        for e in ensemble:
            entropy += e * np.log2(e)
        entropy = -entropy
        return entropy

    def from_mat(file_path):
        mat_data = sio.loadmat(file_path)

        drex_out = mat_data["drex_out"]
        drex_psi = mat_data["drex_psi"]
        input_file_path = mat_data["input_file_path"].item(0)
        input_sequence = mat_data["input_sequence"].T.tolist()[0]

        distribution = drex_out["distribution"].item(0).item(0)
        surprisal = drex_out["surprisal"].item(0)
        joint_surprisal = drex_out["joint_surprisal"].item(0)
        context_beliefs = drex_out["context_beliefs"].item(0)

        change_decision_threshold = mat_data["drex_cd_threshold"].item(0)
        change_decision_changepoint = mat_data["drex_cd"]["changepoint"].item(0).item(0)
        change_decision_probability = mat_data["drex_cd"]["changeprobability"].item(0).T.tolist()[0]
        belief_dynamics = mat_data["drex_bd"].T.tolist()[0]

        # drex_out_prediction_params = drex_out["prediction_params"]

        return DREXOutputParameters(file_path, input_file_path, input_sequence, drex_psi, distribution, surprisal,
                                    joint_surprisal, context_beliefs, change_decision_threshold,
                                    change_decision_changepoint, change_decision_probability, belief_dynamics)


class DREXInstance:
    """This class represents one D-REX instance."""
    def __init__(self, drex_input_parameters, model_io_paths):
        self._matlab_worker = MatlabWorker()
        self._model_io_paths = model_io_paths
        self._drex_input_parameters = drex_input_parameters

    def observe(self, sequence):
        input_file_path = self._model_io_paths["input_file_path"] + ".mat"
        output_file_path = self._model_io_paths["output_file_path"] + ".mat"
        self._drex_input_parameters.with_sequence(sequence).with_output_file_path(output_file_path).write_mat(
            input_file_path)

        result = self._matlab_worker.run_model(input_file_path)
        return result


class DREXInstanceBuilder:
    """This class builds a DREXInstance."""
    def __init__(self, model_io_paths):
        self._model_io_paths = model_io_paths

        self._distribution = DREXDistribution.GAUSSIAN
        self._D = 1
        self._DChanged = False
        self._prior = []
        self._hazard = 0.01
        self._obsnz = 0.0
        self._memory = np.inf
        self._maxhyp = np.inf

    def distribution(self, distribution):  # ToDo remove? since prior determines distribution.
        if distribution == DREXDistribution.GAUSSIAN:
            self._distribution = distribution
            if self._DChanged == False:
                self._D = 1
        elif distribution == DREXDistribution.POISSON:
            self._distribution = distribution
            if self._DChanged == False:
                self._D = 50
        elif distribution in DREXDistribution._member_names_:
            self._distribution = distribution
        else:
            raise ValueError("Invalid distribution!")
        return self

    def D(self, D):
        if D > 0:
            self._D = D
            self._DChanged = True
        else:
            raise ValueError("Invalid D! Value must be greater than 0.")
        return self

    def prior(self, prior):
        if isinstance(prior, DREXPrior):
            self._prior = prior
        else:
            raise ValueError("Invalid prior! Value must be instance of DrexModelPrior.")
        return self

    def hazard(self, hazard):
        if hazard >= 0 and hazard <= 1:
            self._hazard = hazard
        else:
            raise ValueError("Invalid hazard! Value must be >= 0 and <= 1.")
        return self

    def obsnz(self, obsnz):
        if obsnz >= 0 and obsnz <= 1:
            self._obsnz = obsnz
        else:
            raise ValueError("Invalid obsnz! Value must be >= 0 and <= 1.")
        return self

    def memory(self, memory):
        if memory >= 0:
            self._memory = memory
        else:
            raise ValueError("Invalid memory! Value must be >= 0, or numpy.inf (infinite).")
        return self

    def maxhyp(self, maxhyp):
        if maxhyp >= 0:
            self._maxhyp = maxhyp
        else:
            raise ValueError("Invalid maxhyp! Value must be >= 0, or numpy.inf (infinite).")
        return self

    def build(self):
        if isinstance(self._prior, DREXPrior):
            distribution = self._prior.drexDistribution.value
            prior = self._prior.to_dict()
            memory = "Inf" if self._memory == -1 else self._memory
            maxhyp = "Inf" if self._maxhyp == -1 else self._maxhyp

            drex_input_parameters = DREXInputParameters(distribution, self._D, prior, self._hazard, self._obsnz, memory,
                                                        maxhyp)

            return DREXInstance(drex_input_parameters, self._model_io_paths)
        else:
            raise ValueError("Invalid prior! Value must be set.")
