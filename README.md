# **[Kaggle] Cassava Leaf Disease Classification**
Cassava Leaf Disease Classification 
Private 274th | Public 105th
<br>

## **Models**
### **B4**
- 5 folds, input: 512
- CE & LabelSmoothing(0.4) 20 epoch with best epoch threshold
- TTA = 4 (SSR, Hflip, Vflip, Transpose, RandomCrop)
### **SERX101**
- 5 folds, input: 500
- CE & LabelSmoothing(0.1) 10 epoch with best epoch threshold
- LabelSmoothing(0.12) 10 epoch
- TTA = 4 (SSR, Hflip, Vflip, Transpose, CenterCrop, Cutout)
### **VIT-B-16**
- 5 folds, input: 384
- CE & LabelSmoothing(0.1) 10 epoch with best epoch threshold
- TTA = 4 (SSR, Hflip, Vflip, Transpose, RandomCrop)
### **RX50**
- 5 folds, input: 512
- LabelSmoothing(0.12) 10 epoch
- TTA = 4 (SSR, Hflip, Vflip, Transpose, RandomCrop)
<br><br>

## **Ensemble Model**
- Simple Averaing





