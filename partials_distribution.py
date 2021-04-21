from abc import ABC, abstractmethod
from parameters import np, FS
from typing import Union


class PartialsDistribution(ABC):
    def __init__(self, n_partials: int):
        self.n_partials = n_partials
        self.partial_power = None

    @abstractmethod
    def synthesize(self, frequency: Union[int, float], duration: Union[int, float], velocity: int):
        if not (type(frequency) is float or type(frequency) is int):
            raise TypeError("%r should be a float" % frequency)
        if not (type(duration) is float or type(duration) is int):
            raise TypeError("%r should be a float" % duration)
        if not type(velocity) is int:
            raise TypeError("%r should be a int" % velocity)
        if not duration >= 0.:
            raise ValueError("duration should be greater than 0")
        if not 0 <= velocity < 127:
            raise ValueError("velocity should be comprised between 0 and 127")


class AnalyticPartialsDistribution(PartialsDistribution):
    def synthesize(self, *args, **kwargs):
        # ToDo: Implement synthesis
        pass

    def __init__(self, partials_amplitudes, distribution_type=None):
        super().__init__(n_partials=partials_amplitudes.shape[0])
        self.partials_amplitudes = partials_amplitudes
        self.distribution_type = distribution_type


class SyntheticPartialsDistribution(PartialsDistribution):
    def __init__(self, n_partials: int, frequency_evolution: str, time_evolution: str, harmonic: bool = True,
                 frequency_decay_dependency: Union[float, str] = None):
        super().__init__(n_partials)
        self.frequency_evolution = frequency_evolution
        self.time_evolution = time_evolution
        self.harmonic = harmonic

        if frequency_evolution == 'inverse square':
            self.partial_power = 1 / (np.arange(n_partials) + 1)**2
        else:
            raise ValueError("Evolution %r not implemented" % frequency_evolution)

        if time_evolution == 'exponential decay':
            if type(frequency_decay_dependency) is float or type(frequency_decay_dependency) is int:
                if frequency_decay_dependency < 0:
                    Warning("Negative frequency decay implies exponential growth of the amplitude.")
                self.decay = frequency_decay_dependency * np.ones(n_partials)
            else:
                raise ValueError("Frequency decay dependency not implemented")

    def synthesize(self, frequency: Union[int, float], duration: Union[int, float], velocity: int):
        if duration < 0:
            raise ValueError("duration should be positive.")

        t = np.arange(duration * FS) / FS

        if frequency <= 0:
            Warning("Negative frequencies are interpreted as their opposite.")
            frequency = - frequency

        if self.harmonic:
            partials = (velocity / 128) * np.expand_dims(self.partial_power, 1) \
                       * np.sin(2 * np.pi * np.expand_dims((np.arange(self.n_partials) + 1) * frequency, 1) * np.expand_dims(t, 0)) \
                       * np.exp(- 2 * np.pi * np.expand_dims(self.decay, 1) * np.expand_dims(t, 0))
            signal = np.sum(partials, 0)

            return signal
        else:
            raise ValueError("Inharmonic partials distribution not implemented.")


class LinearPartialsDistribution(AnalyticPartialsDistribution):
    def __init__(self, partials_amplitudes, linear_regressions):
        super().__init__(partials_amplitudes, distribution_type="linear")
        self.slopes = linear_regressions[:, 0]
        self.intercept = linear_regressions[:, 1]
        self.rvalue = linear_regressions[:, 2]
        self.pvalue = linear_regressions[:, 3]
        self.stderr = linear_regressions[:, 4]
        self.linear_regressions = linear_regressions
