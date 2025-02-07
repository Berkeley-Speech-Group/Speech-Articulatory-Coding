# Training HiFi-GAN for Articulatory Synthesis


## Installation

Install PyTorch that is compatible with your GPU(s). 

TOOD: Conda

Then, install additional libraries required for model training.
```
pip install -r requirements.txt
```

## Dataset
TODO

```
mkdir data
ln -s YOUR_LIBRITTSR_PATH data/LibriTTS_R
```

### Feature Extraction
```
python extract_features.py --wav_dir=data/LibriTTS_R/wavs --device=cuda:0
```

## Model Training
```
python train_hifigan.py
```

### Exporting Model
TODO