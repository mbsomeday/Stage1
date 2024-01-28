import torch

import basic_learners

MODEL_DICT = {"MyNet": basic_learners.get_MyNet,
              "Inception": basic_learners.get_Inception,
              "ResNet": basic_learners.get_ResNet
              }

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

