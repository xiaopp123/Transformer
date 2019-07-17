import tensorflow as tf
from hparams import Hparams
from model import Transformer
import logging

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
print(hp)

logging.info("# Prepare train/eval batches")

logging.info("# Load model")
m = Transformer(hp)

