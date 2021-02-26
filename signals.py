from midi import midi2piece
from parameters import *
from partials_distribution import SyntheticPartialsDistribution, PartialsDistribution
from samples import SamplesSet


def signal_from_midi(file_name, partials_distribution: PartialsDistribution = None):
    piece = midi2piece(file_name)

    if not partials_distribution:
        partials_distribution = SyntheticPartialsDistribution(n_partials=8, frequency_evolution='inverse square',
                                                              time_evolution='exponential decay', harmonic=True,
                                                              frequency_decay_dependency=0.3)
    samples_set = SamplesSet.from_synthesis(partials_distribution)
    signal = samples_set.synthetize(piece)

    return signal
