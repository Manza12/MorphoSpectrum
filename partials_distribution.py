class PartialsDistribution:
    def __init__(self, partials_amplitudes, distribution_type=None):
        self.partials_amplitudes = partials_amplitudes
        self.number_partials = partials_amplitudes.shape[0]
        self.distribution_type = distribution_type


class LinearPartialsDistribution(PartialsDistribution):
    def __init__(self, partials_amplitudes, linear_regressions):
        super().__init__(partials_amplitudes, distribution_type="linear")
        self.slopes = linear_regressions[:, 0]
        self.intercept = linear_regressions[:, 1]
        self.rvalue = linear_regressions[:, 2]
        self.pvalue = linear_regressions[:, 3]
        self.stderr = linear_regressions[:, 4]
        self.linear_regressions = linear_regressions
