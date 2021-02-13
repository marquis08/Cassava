# from conf import *

import torch 
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import timm




def build_model(args, device):
        
    if args.grayscale: 
        model_list = list()
        model_list.append(nn.Conv2d(1, 3, 1))
        model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes)
        model_list.append(model)
        model = nn.Sequential(*model_list)
    else:
        model = drop_models(args)
        # model_list = list()
        # model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes)
        # model_list.append(model)
        # model = nn.Sequential(*model_list)

    # elif args.model == 'efficientnet_b4':
    #     if args.pretrained and args.mode != 'test':
    #         model = EfficientNet.from_pretrained('efficientnet-b4')
    #     else:
    #         model = EfficientNet.from_name('efficientnet-b4')
    #     in_features = model._fc.in_features
    #     model._fc = nn.Linear(in_features, args.num_classes)
    # elif args.model == 'resnet50':
    #     model = Resnet50(args.num_classes, dropout=False, pretrained=args.pretrained)

    if device: 
        model = model.to(device)
    
    return model

class drop_models(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_list = list()
        # model_list.append(nn.Conv2d(1, 3, 1))
        model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=512)
        model_list.append(model)
        model = nn.Sequential(*model_list)
        self.dropout = nn.Dropout(0.5)                
        self.dropouts = nn.ModuleList([
                    nn.Dropout(0.5) for _ in range(5)])
        self.model = model
        
        self.output_layer = nn.Linear(512, args.num_classes)

    def extract(self, x):
        x=self.model(x)
        return x

    def forward(self, img):
    # def forward(self, img, str_feature):
        #img_feat = self.model(img)
        img_feat = self.extract(img).squeeze(-1).squeeze(-1)
        # str_feat = self.str_model(str_feature)

        # feat = torch.cat([img_feat, str_feat], dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i==0:
                output = self.output_layer(dropout(img_feat))
            else:
                output += self.output_layer(dropout(img_feat))
        else:
            output /= len(self.dropouts)
        #output = self.output_layer(self.dropout(feat))
        # print("OUTPUT: {}".format(output.shape))
        return output