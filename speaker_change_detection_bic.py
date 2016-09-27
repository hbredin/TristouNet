#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

# USAGE:
# export DURATION=2.0  # use 2s sequences
# python speaker_change_detection_bic.py $DURATION

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

from pyannote.audio.segmentation import BICSegmentation
segmentation = BICSegmentation(feature_extractor, covariance_type='full',
                               duration=duration, step=0.100)

# process files from development set
# (and, while we are at it, load groundtruth for later comparison)
predictions = {}
groundtruth = {}
for test_file in protocol.development():
    uri = test_file['uri']
    groundtruth[uri] = test_file['annotation']
    wav = test_file['medium']['wav']
    # this is where the magic happens
    predictions[uri] = segmentation.apply(wav)

# tested thresholds
alphas = np.linspace(0, 1, 50)

# evaluation metrics (purity and coverage)
from pyannote.metrics.segmentation import SegmentationPurity
from pyannote.metrics.segmentation import SegmentationCoverage
purity = [SegmentationPurity() for alpha in alphas]
coverage = [SegmentationCoverage() for alpha in alphas]

# peak detection
from pyannote.audio.signal import Peak
for i, alpha in enumerate(alphas):
    # initialize peak detection algorithm
    peak = Peak(alpha=alpha, min_duration=1.0)
    for uri, reference in groundtruth.items():
        # apply peak detection
        hypothesis = peak.apply(predictions[uri])
        # compute purity and coverage
        purity[i](reference, hypothesis)
        coverage[i](reference, hypothesis)

# print the results in three columns:
# threshold, purity, coverage
TEMPLATE = '{alpha:.3f} {purity:.1f}% {coverage:.1f}%'
for i, a in enumerate(alphas):
    p = 100 * abs(purity[i])
    c = 100 * abs(coverage[i])
    print(TEMPLATE.format(alpha=a, purity=p, coverage=c))
