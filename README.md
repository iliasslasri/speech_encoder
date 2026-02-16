# Simple interface to speech encoders

This repo contains a simple interface to the speech encoders from the Textless NLP
project and [textlesslib library](https://github.com/facebookresearch/textlesslib).

Since fairseq is unmaintained and deprecated, it has become a bit harder to use textlesslib.
This library provides simple classes and functions to use the HuBERT and K-means checkpoints from the project.
It can also be extended to new SSL models and quantizers.

There's not a lot of lines of code here, please have a look to be sure that you understand what is going on.

## Installation

Either clone this repository:
```bash
git clone https://github.com/mxmpl/speech_encoder
cd speech_encoder
uv sync
```

Or install it as a dependency:
```bash
uv pip install git+https://github.com/mxmpl/speech_encoder
# uv add git+https://github.com/mxmpl/speech_encoder
```

### Usage

Basic:
```python
from speech_encoder import SpeechEncoder

encoder = SpeechEncoder.from_textlesslib("hubert-base-ls960", layer=6, vocab_size=200, deduplicate=True)

# Load a single waveform in a tensor of shape (1, T)
waveform = ...
discrete_units = encoder(waveform)
```

Efficient:
```python

from speech_encoder import SpeechEncoder

encoder = SpeechEncoder.from_textlesslib("hubert-base-ls960", layer=6, vocab_size=200, deduplicate=True).cuda()
# Load multiple waveforms and stack them into a single tensor of shape (B, T) with padding.
waveforms = ...
# Create a tensor of shape (B,) with the original length of each waveform (values between 1 and T)
lengths = ...
discrete_units = encoder(waveforms.cuda(), lengths.cuda())
```

Available configurations:
```python
from speech_encoder import SpeechEncoder

print(SpeechEncoder.available_checkpoints())
```

With custom SSL model and quantizer:
```python
from speech_encoder import SpeechEncoder

encoder = SpeechEncoder(ssl, quantizer, deduplicate=True)
```
