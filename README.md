The project contains the experiment which verifies the new method for a microscope image segmentation task. The description of the new method can be found in my [thesis](jndiplom.pdf) (currently only in Czech).

### Requirements
- [Python3](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [OpenCV](https://opencv.org/)
- [SciPy](https://www.scipy.org/)

### Installation
```
conda create --name nucleus python==3.5 tensorflow==1.3.0 numpy opencv scipy
```

### Usage
```
./train.py @settings/border_train
```
```
./predict.py @settings/border_predict
```
