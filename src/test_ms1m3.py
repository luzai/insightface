from recognition.embedding import Embedding
from lz import *

model_path = root_path + 'Evaluation/IJB/pretrained_models/MS1MV2-ResNet100-Arcface/model'
assert os.path.exists(os.path.dirname(model_path)), os.path.dirname(model_path)
gpu_id = 2
embedding = Embedding(model_path, 0, gpu_id)


