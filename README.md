The project contains the experiment which verifies the new method for a microscope image segmentation task. The description of the new method can be found in my [thesis](jndiplom.pdf) or in the [presentation](jan_novacek_segmentace.pdf) (only in Czech).

### Requirements
- [Python3](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [OpenCV](https://opencv.org/)
- [SciPy](https://www.scipy.org/)

### Data
Dataset is possible to download at [kaggle](https://www.kaggle.com/gnovis/nucleus)

### Installation
```
conda create --name nucleus tensorflow numpy opencv scipy
conda activate nucleus
```

### Usage
```
./train.py @settings/border_train
```
```
./predict.py @settings/border_predict
```
