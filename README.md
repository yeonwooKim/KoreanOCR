# KoreanOCR
OCR program for Korean, SNU project 2016

## How to run web server
#### Requirements
* python 3
* flask

```bash
$ python app.py
```

## How to run analysis daemon
#### Requirements
* python 3
* numpy
* scipy
* tensorflow
* hangul-utils
* opencv-python

```bash
$ python server.py
```

Analysis daemon and web server communicates through localhost:1255.

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
