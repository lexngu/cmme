from enum import Enum
import numpy as np
import numpy.typing as npt
from .matlab_worker import MatlabWorker
from abc import ABC, abstractmethod
import scipy.io as sio
import matlab.engine
from pathlib import Path


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
    def nfeatures(self):
        """Returns the number of "features" present in the prior"""
        pass

    @abstractmethod
    def D(self):
        """Returns the used D parameter value"""
        pass

    @abstractmethod
    def to_dict(self):
        """Returns a dictionary representing the prior"""
        pass


class DREXGaussianPrior(DREXPrior):
    """
    Gaussian prior with specific values of mu, ss, n.

    After using D-REX's estimate_suffstat.m function, a 1x1 struct with fields mu, ss, n is returned. Each field contains a 1x1 cell (one for each feature), in which there is a single value containing a column vector with +D+ values.
    """

    def __init__(self, mu, ss, n: list):
        super().__init__(DREXDistribution.GAUSSIAN)

        self.mu = mu  # dim: nfeatures x (D x 1)
        self.ss = ss  # dim: nfeatures x (D x D)
        self.n = n  # dim: nfeatures x 1

    def nfeatures(self):
        return len(self.mu)

    def D(self):
        if isinstance(self.mu[0], matlab.double):
            return self.mu[0].size[0]
        else:
            return 1

    def to_dict(self):
        return {'mu': self.mu, 'ss': self.ss, 'n': self.n}


class DREXLognormalPrior(DREXPrior):
    """Log-scaled Gaussian (=normal) prior with specific values of mu, ss, n."""

    def __init__(self, mu, ss, n: list):
        super().__init__(DREXDistribution.LOGNORMAL)

        self.mu = mu  # dim: nfeatures x (D x 1)
        self.ss = ss  # dim: nfeatures x (D x D)
        self.n = n  # dim: nfeatures x 1

    def nfeatures(self):
        return len(self.mu)

    def D(self):
        if isinstance(self.mu[0], matlab.double):
            return self.mu[0].size[0]
        else:
            return 1

    def to_dict(self):
        return {'mu': self.mu, 'ss': self.ss, 'n': self.n}


class DREXGmmPrior(DREXPrior):
    """Gaussian Mixture Model (GMM) prior"""

    def __init__(self,
                 mu: matlab.double,
                 sigma: matlab.double,
                 n: matlab.double,
                 pi: matlab.double,
                 sp: matlab.double,
                 k: matlab.double,
                 D: matlab.double):
        super().__init__(DREXDistribution.GMM)

        self.mu = mu  # dim: nfeatures x max_ncomp
        self.sigma = sigma  # dim: nfeatures x max_ncomp
        self.n = n  # dim: nfeatures x max_ncomp
        self.pi = pi
        self.sp = sp
        self.k = k
        self._D = D
        # ToDo add parameter: beta

    def nfeatures(self):
        return len(self.mu)

    def D(self):
        return self._D

    def max_ncomp(self):
        return self.mu[0].size[1]

    def to_dict(self):
        return {'mu': self.mu, 'sigma': self.sigma, 'n': self.n, 'pi': self.pi, 'sp': self.sp, 'k': self.k}


class DREXPoissonPrior(DREXPrior):
    """Poisson prior with specific values of lambda and n."""

    def __init__(self,
                 lambd: matlab.double,
                 n: matlab.double,
                 D: matlab.double):
        super().__init__(DREXDistribution.POISSON)

        self.lambd = lambd  # dim: nfeatures x nfeatures
        self.n = n  # dim: nfeatures x 1
        self._D = D  # dim: 1

    def nfeatures(self):
        return len(self.n)

    def D(self):
        return self._D

    def to_dict(self):
        return {'lambda': self.lambd, 'n': self.n}


class DREXPriorInputParameters:
    """This class contains instructions for D-REX's estimate_suffstat.m function, which outputs a file that is parsed
    by DREXPriorOutputParameters. """

    def __init__(self,
                 distribution: DREXDistribution,
                 sequence: npt.NDArray,
                 D,
                 maxNComp=None):
        if (distribution == DREXDistribution.GMM and maxNComp == None):
            raise ValueError("maxNComp is invalid! If distribution == GMM, maxNComp needs to be set.")
        if (distribution != DREXDistribution.GMM and maxNComp != None):
            raise ValueError("maxNComp is invalid! If distribution != GMM, maxNComp must not be set.")

        self._distribution = distribution
        self._D = D
        self._maxNComp = maxNComp
        self._sequence = sequence.reshape(sequence.shape[0], sequence.shape[1], 1)  # dim: time x feature x 1(=trial)
        self._sequence = np.moveaxis(self._sequence, 1, 2)
        self._sequence = np.moveaxis(self._sequence, 0, 2)
        self._sequence = np.moveaxis(self._sequence, 0, 1)  # dim: trial x time x feature

    def with_results_file_path(self, results_file_path):
        self._results_file_path = results_file_path
        return self

    def write_mat(self, file_path: Path):
        mat_data = {
            "xs": self._sequence,
            "params": {
                "distribution": self._distribution.value,
                "D": self._D
            },
            "results_file_path": str(self._results_file_path)
        }
        if self._distribution == DREXDistribution.GMM:
            mat_data["params"]["max_ncomp"] = self._maxNComp

        sio.savemat(str(file_path), mat_data)

        return file_path


class DREXPriorOutputParameters:
    """This class parses the output of D-REX's estimate_suffstat.m function and returns a DREXDistribution."""

    def __init__(self):
        pass

    def from_mat(file_path):
        eng = matlab.engine.start_matlab()
        mat_data = eng.load(file_path)
        data_set = mat_data["drex_out"]
        distribution = DREXDistribution(mat_data["distribution"])
        columns = data_set.keys()
        data = {c: data_set[c] for c in columns}

        if distribution == DREXDistribution.GAUSSIAN:
            result = DREXGaussianPrior(data['mu'], data['ss'], data['n'])
        elif distribution == DREXDistribution.LOGNORMAL:
            result = DREXLognormalPrior(data['mu'], data['ss'], data['n'])
        elif distribution == DREXDistribution.GMM:
            result = DREXGmmPrior(data['mu'], data['sigma'], data['n'], data['pi'], data['sp'], data['k'],
                                  mat_data['D'])
        elif distribution == DREXDistribution.POISSON:
            result = DREXPoissonPrior(data['lambda'], data['n'], mat_data['D'])

        return result


class DREXPriorEstimator:
    """This class triggers the D-REX's prior calculation."""

    def __init__(self, instructions_file_path: Path, results_file_path: Path):
        self.instructions_file_path = instructions_file_path
        self.results_file_path = results_file_path

        self._matlab_worker = MatlabWorker()

    def estimate(self, distribution: DREXDistribution, sequence, D, maxNComp=None):
        """Given the desired distribution and the input sequence, D-REX's estimate_suffstat.m function is triggered."""

        if len(sequence) == 0:
            raise ValueError("sequence is empty! should contain at least one element.")
        if isinstance(sequence[0], int):
            input_sequence = np.array([sequence], dtype=float)  # dim: feature x time
        elif isinstance(sequence[0], list):
            input_sequence = np.array(sequence, dtype=float)  # dim: feature x time
        else:
            raise ValueError("sequence is invalid!")

        DREXPriorInputParameters(distribution, input_sequence, D, maxNComp).with_results_file_path(
            str(self.results_file_path)).write_mat(
            self.instructions_file_path)

        return self._matlab_worker.estimate_prior(self.instructions_file_path)


class DREXInputParameters:
    """This class represents all of D-REX's input parameters."""

    def __init__(self, prior: DREXPrior, hazard, obsnz: list, memory, maxhyp):
        self._prior = prior
        self._hazard = hazard
        self._obsnz = obsnz
        self._memory = memory
        self._maxhyp = maxhyp
        self._sequence = None

        self._results_file_path = None

    def with_sequence(self, sequence):
        self._sequence = matlab.double(sequence)
        return self

    def with_results_file_path(self, results_file_path: Path):
        self._results_file_path = results_file_path
        return self

    def write_mat(self, file_path: Path):
        engs = matlab.engine.find_matlab()
        if len(engs) > 0:
            eng = matlab.engine.connect_matlab(engs[0])
        else:
            eng = matlab.engine.start_matlab()

        mat_data = {
            "x": eng.transpose(self._sequence),
            "results_file_path": str(self._results_file_path),
            "params": {
                "distribution": self._prior.drexDistribution.value,
                "D": eng.double(self._prior.D()),
                "prior": self._prior.to_dict(),
                "hazard": self._hazard,
                "obsnz": matlab.double(self._obsnz),
                "memory": self._memory,
                "maxhyp": self._maxhyp,
            }
        }
        if self._prior.drexDistribution == DREXDistribution.GMM:
            mat_data["params"]["max_ncomp"] = self._prior.max_ncomp()

        for k, v in mat_data.items():
            eng.workspace[k] = v
        keys = mat_data.keys()
        eng.save(str(file_path), *keys, nargout=0)

        # sio.savemat(filename, mat_data)

        return file_path

    def read_csv(filename):
        return DREXInputParameters()


class DREXOutputParameters:
    """This class represents all of D-REX's output parameters."""

    def __init__(self, source_file_path, instructions_file_path, input_sequence, psi, distribution, surprisal, joint_surprisal,
                 context_beliefs, change_decision_threshold, change_decision_changepoint, change_decision_probability,
                 belief_dynamics):
        self.source_file_path = source_file_path
        self.instructions_file_path = instructions_file_path
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
        instructions_file_path = mat_data["instructions_file_path"].item(0)
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

        return DREXOutputParameters(file_path, instructions_file_path, input_sequence, drex_psi, distribution, surprisal,
                                    joint_surprisal, context_beliefs, change_decision_threshold,
                                    change_decision_changepoint, change_decision_probability, belief_dynamics)


class DREXInstance:
    """This class represents one D-REX instance."""

    def __init__(self, drex_input_parameters: DREXInputParameters, instructions_file_path: Path, results_file_path: Path):
        self._matlab_worker = MatlabWorker()

        self._drex_input_parameters = drex_input_parameters
        self.instructions_file_path = instructions_file_path
        self.results_file_path = results_file_path

    def observe(self, sequence: list):
        self._drex_input_parameters.with_sequence(sequence).with_results_file_path(self.results_file_path).write_mat(
            self.instructions_file_path)

        result = self._matlab_worker.run_model(self.instructions_file_path)
        return result


class DREXInstanceBuilder:
    """This class builds a DREXInstance."""

    def __init__(self, prior):
        if isinstance(prior, DREXPrior):
            self._prior = prior
            self._nfeatures = self._prior.nfeatures()
        else:
            raise ValueError("Invalid prior! Value must be instance of DrexModelPrior.")

        self._D = self._prior.D()
        self._DChanged = False
        self._hazard = 0.01
        self._obsnz = [0.0] * self._nfeatures
        self._memory = np.inf
        self._maxhyp = np.inf

        self.instructions_file_path = None
        self.results_file_path = None

    def hazard(self, hazard):
        if hazard >= 0 and hazard <= 1:
            self._hazard = hazard
        else:
            raise ValueError("Invalid hazard! Value must be >= 0 and <= 1.")
        return self

    def obsnz(self,
              obsnz: list):
        """For each feature: observation noise"""
        if len(obsnz) == self._nfeatures:
            self._obsnz = obsnz
        else:
            raise ValueError("Invalid obsnz! Must be list of length = {}".format(self._nfeatures))
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

    def with_instructions_file_path(self, instructions_file_path: Path):
        self.instructions_file_path = instructions_file_path

        return self

    def with_results_file_path(self, results_file_path: Path):
        self.results_file_path = results_file_path

        return self

    def build(self):
        if self.instructions_file_path is None:
            raise Exception("instructions_file_path required!")
        if self.results_file_path is None:
            raise Exception("results_file_path required!")

        memory = "Inf" if self._memory == -1 else self._memory
        maxhyp = "Inf" if self._maxhyp == -1 else self._maxhyp

        drex_input_parameters = DREXInputParameters(self._prior, self._hazard, self._obsnz, memory,
                                                    maxhyp)

        return DREXInstance(drex_input_parameters, self.instructions_file_path, self.results_file_path)
