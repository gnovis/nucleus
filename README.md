The project contains the experiment which verifies the new method for a microscope image segmentation task. The description of the new method can be found in my [thesis](jndiplom.pdf) (Czech) or in the [presentation](nuclei_seg_jn.pdf) (English).

![](in_out.png?raw=true "CNN input and output examples")
CNN (Encoder-Decoder) takes cell nuclei cluster with incomplete border as an input and produces full border as an output.

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
