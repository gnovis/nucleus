from .dataset import DataSet
from .net import EncoderDecoderNet

NETS = {
    "EncoderDecoderNet": EncoderDecoderNet,
}
DATASETS = {
    "DataSet": DataSet,
}
