from conf import *
import torch.nn as nn
from utils import SCELoss, LabelSmoothingLoss, bi_tempered_logistic_loss

def fetch_loss(args):
    if args.loss_fn == "SCE":
        return SCELoss()
    elif args.loss_fn == "CE":
        return nn.CrossEntropyLoss()
    elif args.loss_fn == "Label":
        return LabelSmoothingLoss(classes=args.num_classes, smoothing=args.label_smoothing_ratio)
    elif args.loss_fn == "BTLL":
        return bi_tempered_logistic_loss(t1=0.2, t2=1.0) # Large parameter --> t1=0.2, t2=1.0
    else:
        NotImplementedError
