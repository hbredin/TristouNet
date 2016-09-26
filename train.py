# USAGE:
# export DURATION=2.0  # use 2s sequences
# python train.py $DURATION

# ---- <edit> -----------------------------------------------------------------
# environment
WAV_TEMPLATE = '/path/to/where/files/are/stored/{uri}.wav'
LOG_DIR = '/path/to/where/trained/models/are/stored'
# ---- </edit> ---------------------------------------------------------------

# sequence duration (in seconds)
import sys
duration = float(sys.argv[1])

# number of epoch
nb_epoch = 1000

LOG_DIR = LOG_DIR + '/{duration:.1f}s'.format(duration=duration)

import numpy as np
np.random.seed(1337)  # for reproducibility

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

# TristouNet architecture
from pyannote.audio.embedding.models import TristouNet
architecture = TristouNet()

# triplet loss
from pyannote.audio.embedding.losses import TripletLoss
margin = 0.2    # `alpha` in the paper
loss = TripletLoss(architecture, margin=margin)

from pyannote.audio.embedding.base import SequenceEmbedding
embedding = SequenceEmbedding(
    loss=loss, optimizer='rmsprop', log_dir=LOG_DIR)

# triplet sampling
# this might take some time as the whole corpus is loaded in memory,
# and the whole set of MFCC features sequences is precomputed
from pyannote.audio.embedding.generator import TripletBatchGenerator
per_label = 40  # `n` in the paper
batch_size = 8192
generator = TripletBatchGenerator(
    feature_extractor, protocol.train(), embedding, margin=margin,
    duration=duration, per_label=per_label, batch_size=batch_size)

# shape of feature sequences (n_frames, n_features)
input_shape = generator.get_shape()

# number of samples per epoch
# (rounded to closest batch_size multiple)
samples_per_epoch = per_label * (per_label - 1) * generator.n_labels
samples_per_epoch = samples_per_epoch - (samples_per_epoch % batch_size)

# actual training
embedding.fit(input_shape, generator, samples_per_epoch, nb_epoch)
