# USAGE:
# export DURATION=2.0  # use 2s sequences
# python same_different_experiment_baseline.py $DURATION

# ---- <edit> -----------------------------------------------------------------
# environment
WAV_TEMPLATE = '/path/to/where/files/are/stored/{uri}.wav'
LOG_DIR = '/path/to/where/trained/models/are/stored'
# ---- </edit> ---------------------------------------------------------------

# sequence duration (in seconds)
import sys
duration = float(sys.argv[1])

LOG_DIR = LOG_DIR + '/{duration:.1f}s'.format(duration=duration)

import numpy as np
np.random.seed(1337)  # for reproducibility

# feature extraction
from pyannote.audio.features.yaafe import YaafeMFCC
feature_extractor = YaafeMFCC(e=False, De=False, DDe=False,
                              coefs=11, D=False, DD=False)

# ETAPE database
medium_template = {'wav': WAV_TEMPLATE}
from pyannote.database import Etape
database = Etape(medium_template=medium_template)

# experimental protocol (ETAPE TV subset)
protocol = database.get_protocol('SpeakerDiarization', 'TV')

batch_size = 32

# generate set of labeled sequences
from pyannote.audio.generators.labels import \
    LabeledFixedDurationSequencesBatchGenerator
generator = LabeledFixedDurationSequencesBatchGenerator(
    feature_extractor, duration=duration, step=duration, batch_size=-1)
X, y = zip(*generator(protocol.development()))
X, y = np.vstack(X), np.hstack(y)

# randomly select (at most) 100 sequences from each speaker to ensure
# all speakers have the same importance in the evaluation
unique, y, counts = np.unique(y, return_inverse=True, return_counts=True)
n_speakers = len(unique)
indices = []
for speaker in range(n_speakers):
    i = np.random.choice(np.where(y == speaker)[0], size=min(100, counts[speaker]), replace=False)
    indices.append(i)
indices = np.hstack(indices)
X, y = X[indices], y[indices, np.newaxis]

# one Gaussian per segment
from pyannote.algorithms.stats.gaussian import Gaussian
full = [Gaussian(covariance_type='full').fit(x) for x in X]
diag = [Gaussian(covariance_type='diag').fit(x) for x in X]

# compute BIC and divergence between every pair of sequences
import itertools
n_sequences = len(X)
bic = np.zeros((n_sequences, n_sequences), dtype=np.float32)
div = np.zeros((n_sequences, n_sequences), dtype=np.float32)

for i, j in itertools.combinations(range(n_sequences), 2):

    gi, gj = full[i], full[j]
    bic[i, j] = gi.bic(gj, penalty_coef=0)[0]
    bic[j ,i] = bic[i, j]

    gi, gj = diag[i], diag[j]
    div[i, j] = gi.divergence(gj)
    div[j, i] = div[i, j]

from scipy.spatial.distance import squareform
bic = squareform(bic, checks=False)
div = squareform(div, checks=False)

# compute same/different groundtruth
from scipy.spatial.distance import pdist
y_true = pdist(y, metric='chebyshev') < 1

# plot positive/negative scores distribution
# plot DET curve and return equal error rate
from pyannote.metrics.plot.binary_classification import \
    plot_det_curve, plot_distributions

bic_prefix = LOG_DIR + '/plot.bic'
plot_distributions(y_true, bic, bic_prefix, xlim=(0, 2), ymax=3, nbins=100)
eer = plot_det_curve(y_true, -bic, bic_prefix)
print('BIC EER = {eer:.2f}%'.format(eer=100*eer))

div_prefix = LOG_DIR + '/plot.div'
plot_distributions(y_true, div, div_prefix, xlim=(0, 2), ymax=3, nbins=100)
eer = plot_det_curve(y_true, -div, div_prefix)
print('DIV EER = {eer:.2f}%'.format(eer=100*eer))
