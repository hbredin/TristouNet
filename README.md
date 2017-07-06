
This repository is no longer supported: a complete rewrite (leading to much better performance) is now part of [`pyannote-audio`](https://github.com/pyannote/pyannote-audio), which provides:

* a [tutorial](https://github.com/pyannote/pyannote-audio#tutorials) to train, validate, and apply a TristouNet embedding
* a pre-trained TristouNet model 

# TristouNet: Triplet Loss for Speaker Turn Embedding

Code for http://arxiv.org/abs/1609.04301

## Citation

```bibtex
@inproceedings{Bredin2017,
    author = {Herv\'{e} Bredin},
    title = {{TristouNet: Triplet Loss for Speaker Turn Embedding}},
    booktitle = {42nd IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017},
    year = {2017},
    url = {http://arxiv.org/abs/1609.04301},
}
```

## Installation

**Foreword:** Sorry about the `python=2.7` constraint but [`yaafe`](https://github.com/Yaafe/Yaafe) does not support Python 3 at the time of writing this.  
Ping me if this is no longer true!

```bash
$ conda create --name tristounet python=2.7 anaconda
$ source activate tristounet
$ conda install gcc
$ conda install -c yaafe yaafe=0.65
$ pip install "theano==0.8.2"
$ pip install "keras==1.1.0"
$ pip install "pyannote.metrics==0.9"
$ pip install "pyannote.audio==0.1.4"
$ pip install "pyannote.db.etape==0.2.1"
$ pip install "pyannote.database==0.4.1"
$ pip install "pyannote.generators==0.1.1"
```

Using `pip freeze`, make sure you have installed these exact versions before submitting an issue.

What did I just install?

- [`keras`](keras.io) (and its [`theano`](http://deeplearning.net/software/theano/) backend) is used for all things deep.
- [`yaafe`](https://github.com/Yaafe/Yaafe) is used for MFCC feature extraction in [`pyannote.audio`](http://pyannote.github.io).
  You might also want to checkout [`librosa`](http://librosa.github.io) (easy to install, but much slower) though [`pyannote.audio`](http://pyannote.github.io) does not support it yet.
- [`pyannote.audio`](http://pyannote.github.io) is where the magic happens (TristouNet architecture, triplet loss, and triplet sampling)
- [`pyannote.db.etape`](http://pyannote.github.io) is the ETAPE plugin for [`pyannote.database`](http://pyannote.github.io), a common API for multimedia databases and experimental protocols (*e.g.* `train`/`dev`/`test` sets definition).
- [`pyannote.metrics`](http://pyannote.github.io) provides evaluation metrics.

Then, edit `~/.keras/keras.json` to configure `keras` with `theano` backend.

```json
$ cat ~/.keras/keras.json
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

#### About the ETAPE database

To reproduce the experiment, you obviously need to have access to the ETAPE corpus.  
It can be obtained from [ELRA catalogue](http://islrn.org/resources/425-777-374-455-4/).

However, if you own another corpus with *"who speaks when"* annotations, then
you are just one step away from testing `TristouNet` on your own corpus.

Just fork [`pyannote.db.etape`](http://github.com/pyannote/pyannote-db-etape)
and adapt the code to your own database. If (and only if) the database is available for anyone to use, I am willing to help you do that.

## Training

For convenience, this script is also available in `train.py`.
See file header for usage instructions.

```python
# environment
>>> WAV_TEMPLATE = '/path/to/where/files/are/stored/{uri}.wav'
>>> LOG_DIR = '/path/to/where/trained/models/are/stored'

# for reproducibility
>>> import numpy as np
>>> np.random.seed(1337)

# feature extraction
>>> from pyannote.audio.features.yaafe import YaafeMFCC
>>> feature_extractor = YaafeMFCC(e=False, De=True, DDe=True,
...                               coefs=11, D=True, DD=True)

# ETAPE database
>>> medium_template = {'wav': WAV_TEMPLATE}
>>> from pyannote.database import Etape
>>> database = Etape(medium_template=medium_template)

# experimental protocol (ETAPE TV subset)
>>> protocol = database.get_protocol('SpeakerDiarization', 'TV')

# TristouNet architecture
>>> from pyannote.audio.embedding.models import TristouNet
>>> architecture = TristouNet()

# triplet loss
>>> from pyannote.audio.embedding.losses import TripletLoss
>>> margin = 0.2    # `alpha` in the paper
>>> loss = TripletLoss(architecture, margin=margin)

>>> from pyannote.audio.embedding.base import SequenceEmbedding
>>> log_dir = LOG_DIR
>>> embedding = SequenceEmbedding(
...     loss=loss, optimizer='rmsprop', log_dir=log_dir)

# sequence duration (in seconds)
>>> duration = 2.0

# triplet sampling
# this might take some time as the whole corpus is loaded in memory,
# and the whole set of MFCC features sequences is precomputed
>>> from pyannote.audio.embedding.generator import TripletBatchGenerator
>>> per_label = 40  # `n` in the paper
>>> batch_size = 32
>>> generator = TripletBatchGenerator(
...     feature_extractor, protocol.train(), embedding, margin=margin,
...     duration=duration, per_label=per_label, batch_size=batch_size)

# shape of feature sequences (n_frames, n_features)
>>> input_shape = generator.get_shape()

# number of samples per epoch
# (rounded to closest batch_size multiple)
>>> samples_per_epoch = per_label * (per_label - 1) * generator.n_labels
>>> samples_per_epoch = samples_per_epoch - (samples_per_epoch % batch_size)

# number of epochs
>>> nb_epoch = 50

# actual training
>>> embedding.fit(input_shape, generator, samples_per_epoch, nb_epoch)
```

Here is an excerpt of the expected output:

```
UserWarning: 68 labels (out of 179) have less than 40 training samples.

Epoch 1/1000
278528/278528 [==============================] - 1722s - loss: 0.1074
Epoch 2/1000
278528/278528 [==============================] - 1701s - loss: 0.1179
Epoch 3/1000
278528/278528 [==============================] - 1691s - loss: 0.1204
...
Epoch 48/1000
278528/278528 [==============================] - 1688s - loss: 0.0925
Epoch 49/1000
278528/278528 [==============================] - 1675s - loss: 0.0937
Epoch 50/1000
278528/278528 [==============================] - 1678s - loss: 0.0863
```

## *"same/different"* toy experiment

For convenience, this script is also available in `same_different_experiment.py`.
See file header for usage instructions.

```python
# for reproducibility
>>> import numpy as np
>>> np.random.seed(1337)

# generate set of labeled sequences
>>> from pyannote.audio.generators.labels import \
...     LabeledFixedDurationSequencesBatchGenerator
>>> generator = LabeledFixedDurationSequencesBatchGenerator(
...     feature_extractor, duration=duration, step=duration, batch_size=-1)
>>> X, y = zip(*generator(protocol.development()))
>>> X, y = np.vstack(X), np.hstack(y)

# randomly select (at most) 100 sequences from each speaker to ensure
# all speakers have the same importance in the evaluation
>>> unique, y, counts = np.unique(y, return_inverse=True, return_counts=True)
>>> n_speakers = len(unique)
>>> indices = []
>>> for speaker in range(n_speakers):
...     i = np.random.choice(np.where(y == speaker)[0], size=min(100, counts[speaker]), replace=False)
...     indices.append(i)
>>> indices = np.hstack(indices)
>>> X, y = X[indices], y[indices, np.newaxis]

# load pre-trained embedding
>>> architecture_yml = log_dir + '/architecture.yml'
>>> weights_h5 = log_dir + '/weights/{epoch:04d}.h5'.format(epoch=nb_epoch - 1)
>>> embedding = SequenceEmbedding.from_disk(architecture_yml, weights_h5)

# embed all sequences
>>> fX = embedding.transform(X, batch_size=batch_size, verbose=0)

# compute euclidean distance between every pair of sequences
>>> from scipy.spatial.distance import pdist
>>> distances = pdist(fX, metric='euclidean')

# compute same/different groundtruth
>>> y_true = pdist(y, metric='chebyshev') < 1

# plot positive/negative scores distribution
# plot DET curve and return equal error rate
>>> from pyannote.metrics.plot.binary_classification import \
...     plot_det_curve, plot_distributions
>>> prefix = log_dir + '/plot.{epoch:04d}'.format(epoch=nb_epoch - 1)
>>> plot_distributions(y_true, distances, prefix, xlim=(0, 2), ymax=3, nbins=100)
>>> eer = plot_det_curve(y_true, -distances, prefix)
>>> print('EER = {eer:.2f}%'.format(eer=100*eer))
```

This is the expected output:

```
EER = 14.44%
```

Baseline results with BIC and Gaussian divergence on the same experimental
protocol can be obtained with `same_different_experiment_baseline.py`.
See file header for usage instructions. This is the expected output:

```
BIC EER = 20.53%
DIV EER = 22.49%
```

## Speaker change detection

For convenience, this script is also available in `speaker_change_detection.py`.
See file header for usage instructions.

```python
# for reproducibility
>>> import numpy as np
>>> np.random.seed(1337)

# load pre-trained embedding
>>> architecture_yml = log_dir + '/architecture.yml'
>>> weights_h5 = log_dir + '/weights/{epoch:04d}.h5'.format(epoch=nb_epoch - 1)
>>> embedding = SequenceEmbedding.from_disk(architecture_yml, weights_h5)

>>> from pyannote.audio.embedding.segmentation import Segmentation
>>> segmentation = Segmentation(embedding, feature_extractor,
...                             duration=duration, step=0.100)

# process files from development set
# (and, while we are at it, load groundtruth for later comparison)
>>> predictions = {}
>>> groundtruth = {}
>>> for test_file in protocol.development():
...     uri = test_file['uri']
...     groundtruth[uri] = test_file['annotation']
...     wav = test_file['medium']['wav']
...     # this is where the magic happens
...     predictions[uri] = segmentation.apply(wav)

# tested thresholds
>>> alphas = np.linspace(0, 1, 50)

# evaluation metrics (purity and coverage)
>>> from pyannote.metrics.segmentation import SegmentationPurity
>>> from pyannote.metrics.segmentation import SegmentationCoverage
>>> purity = [SegmentationPurity() for alpha in alphas]
>>> coverage = [SegmentationCoverage() for alpha in alphas]

# peak detection
>>> from pyannote.audio.signal import Peak
>>> for i, alpha in enumerate(alphas):
...     # initialize peak detection algorithm
...     peak = Peak(alpha=alpha, min_duration=1.0)
...     for uri, reference in groundtruth.items():
...         # apply peak detection
...         hypothesis = peak.apply(predictions[uri])
...         # compute purity and coverage
...         purity[i](reference, hypothesis)
...         coverage[i](reference, hypothesis)

# print the results in three columns:
# threshold, purity, coverage
>>> TEMPLATE = '{alpha:.2f} {purity:.1f}% {coverage:.1f}%'
>>> for i, a in enumerate(alphas):
...     p = 100 * abs(purity[i])
...     c = 100 * abs(coverage[i])
...     print(TEMPLATE.format(alpha=a, purity=p, coverage=c))
```

Here is an excerpt of the expected result:

```
0.00 95.2% 38.9%
0.02 95.2% 38.9%
0.04 95.2% 39.0%
...
0.49 94.4% 55.0%
...
0.96 86.7% 93.7%
0.98 86.3% 94.4%
1.00 86.2% 94.9%
```

Replace `Segmentation` by `BICSegmentation` or `GaussianDivergenceSegmentation`
to get baseline results. You should also remove derivatives from MFCC feature
extractor (`D=False, DD=False, De=False, DDe=False`) as it leads to better
baseline performance. You might need to install `pyannote.algorithms==0.6.5` first.

```python
>>> from pyannote.audio.segmentation import BICSegmentation
>>> segmentation = BICSegmentation(feature_extractor,
...                                covariance_type='full',
...                                duration=duration,
...                                step=0.100)
```

Here is an excerpt of the expected result:

```
0.000 95.1% 37.7%
0.020 95.1% 37.7%
0.041 95.1% 37.7%
...
0.388 94.4% 48.0%
...
0.959 86.4% 86.8%
0.980 86.2% 88.0%
1.000 85.8% 88.8%
```

```python
>>> from pyannote.audio.segmentation import GaussianDivergenceSegmentation
>>> segmentation = GaussianDivergenceSegmentation(feature_extractor,
...                                               duration=duration,
...                                               step=0.100)
```

Here is an excerpt of the expected result:


```
0.000 95.2% 36.2%
0.020 95.2% 36.2%
...
0.327 94.4% 48.3%
...
0.959 87.5% 85.8%
0.980 87.4% 86.5%
1.000 87.0% 87.5%
```

For convenience, two scripts are available: `speaker_change_detection_bic.py`
and `speaker_change_detection_div.py`. See file header for usage instructions.
