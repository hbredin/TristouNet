# ---- <edit> -----------------------------------------------------------------
# environment
WAV_TEMPLATE = '/path/to/where/files/are/stored/{uri}.wav'
LOG_DIR = '/path/to/where/trained/models/are/stored'

# sequence duration (in seconds)
duration = 2.0

# number of epoch
nb_epoch = 70
# ---- </edit> ---------------------------------------------------------------

# feature extraction
from pyannote.audio.features.yaafe import YaafeMFCC
feature_extractor = YaafeMFCC(e=False, De=True, DDe=True,
                              coefs=11, D=True, DD=True)

# ETAPE database
medium_template = {'wav': WAV_TEMPLATE}
from pyannote.database import Etape
database = Etape(medium_template=medium_template)

# experimental protocol (ETAPE TV subset)
protocol = database.get_protocol('SpeakerDiarization', 'TV')

from pyannote.audio.embedding.base import SequenceEmbedding

batch_size = 32

# generate set of labeled sequences
import numpy as np
from pyannote.audio.generators.labels import \
    LabeledFixedDurationSequencesBatchGenerator
generator = LabeledFixedDurationSequencesBatchGenerator(
    feature_extractor, duration=duration, step=duration, batch_size=-1)
X, y = zip(*generator(protocol.development()))
X, y = np.vstack(X), np.hstack(y)

# make random 'deterministic'
np.random.seed(1337)

# randomly select 20 sequences from each speaker to ensure
# all speakers have the same importance in the evaluation
unique, y, counts = np.unique(y, return_inverse=True, return_counts=True)
n_speakers = len(unique)
indices = []
for speaker in range(n_speakers):
    i = np.random.choice(np.where(y == speaker)[0], size=20, replace=True)
    indices.append(i)
indices = np.hstack(indices)
X, y = X[indices], y[indices, np.newaxis]

# load pre-trained embedding
architecture_yml = LOG_DIR + '/architecture.yml'
weights_h5 = LOG_DIR + '/weights/{epoch:04d}.h5'.format(epoch=nb_epoch - 1)
embedding = SequenceEmbedding.from_disk(architecture_yml, weights_h5)

# embed all sequences
fX = embedding.transform(X, batch_size=batch_size, verbose=0)

# compute euclidean distance between every pair of sequences
from scipy.spatial.distance import pdist
distances = pdist(fX, metric='euclidean')

# compute same/different groundtruth
y_true = pdist(y, metric='chebyshev') < 1

# plot positive/negative scores distribution
# plot DET curve and return equal error rate
from pyannote.metrics.plot.binary_classification import \
    plot_det_curve, plot_distributions
prefix = LOG_DIR + '/plot.{epoch:04d}'.format(epoch=nb_epoch - 1)
plot_distributions(y_true, distances, prefix, xlim=(0, 2), ymax=3, nbins=100)
eer = plot_det_curve(y_true, -distances, prefix)
print('EER = {eer:.2f}%'.format(eer=100*eer))
