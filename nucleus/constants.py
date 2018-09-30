from .dataset import DataSet
from .net import EncoderDecoderNet, SelectionNet, NoPoolNet, EncoderDecoderDoubleNet

NETS = {
    "SelectionNet": SelectionNet,
    "EncoderDecoderNet": EncoderDecoderNet,
    "NoPoolNet": NoPoolNet,
    "EncoderDecoderDoubleNet": EncoderDecoderDoubleNet,
}
DATASETS = {
    "DataSet": DataSet,
}
