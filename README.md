# Semi-Supervised Medical Image Segmentation through Label-Driven Space Structure Augmentation
by Zhijun Yan, Yonghong Hou, and Pengyu Zhao.
## Introduction
Official code for "Semi-Supervised Medical Image Segmentation through Label-Driven Space Structure Augmentation"
## Requirements
This repository is based on PyTorch 2.0.0, CUDA 11.4 and Python 3.9.15 All experiments in our paper were conducted on TianX GPUs with an identical experimental setting.
## Usage
1. We provide `code`, `data_split` and `models` for LA and ACDC dataset.

2. Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data) and [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC).

   Put the LA dataset in `data_split/LA`
  
   Put the ACDC dataset in `data_split/ACDC/data`

3. To train a model,
```
python ./code/LA_train_SCCS.py  #for LA training
python ./code/ACDC_train_SCCS.py  #for ACDC training
``` 

4. To test a model,
```
python ./code/test_LA.py  #for LA testing
python ./code/testacdc.py  #for ACDC testing
```


## Acknowledgements
Our code is largely based on [BCP](https://github.com/DeepMed-Lab-ECNU/BCP). Thanks for these authors for their valuable work, hope our work can also contribute to related research.

## Questions
If you have any questions, welcome contact me at 'yan_zj@tju.edu.cn'



