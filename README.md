# KoreanOCR
OCR program for Korean, SNU project 2016

## How to run server
#### Requirements
* python 3
* flask
* numpy
* scipy
* tensorflow
* hangul-utils
* opencv-python

```bash
$ python app.py
```

## How to train
#### Requirements
* python 3.4
* numpy
* scipy
* tensorflow
* hangul-utils

#### Arguments
```
python trainer.py
-i <tar data>
-o <path to save>
-e <epoch to train>
-b <mini batch size>
```

By default, 4 epochs and 100 per batch.

#### Example
```bash
$ python trainer.py -i data/tar/161020.tgz -o data/ckpt/temp.ckpt
```