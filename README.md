# M<sup>3</sup>Net: Multilevel, Mixed and Multistage Attention Network for Salient Object Detection
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m-3-net-multilevel-mixed-and-multistage/salient-object-detection-on-dut-omron)](https://paperswithcode.com/sota/salient-object-detection-on-dut-omron?p=m-3-net-multilevel-mixed-and-multistage)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m-3-net-multilevel-mixed-and-multistage/salient-object-detection-on-ecssd)](https://paperswithcode.com/sota/salient-object-detection-on-ecssd?p=m-3-net-multilevel-mixed-and-multistage)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m-3-net-multilevel-mixed-and-multistage/salient-object-detection-on-hku-is)](https://paperswithcode.com/sota/salient-object-detection-on-hku-is?p=m-3-net-multilevel-mixed-and-multistage)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m-3-net-multilevel-mixed-and-multistage/salient-object-detection-on-duts-te)](https://paperswithcode.com/sota/salient-object-detection-on-duts-te?p=m-3-net-multilevel-mixed-and-multistage)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/m-3-net-multilevel-mixed-and-multistage/salient-object-detection-on-pascal-s)](https://paperswithcode.com/sota/salient-object-detection-on-pascal-s?p=m-3-net-multilevel-mixed-and-multistage)

Source code of 'M<sup>3</sup>Net: Multilevel, Mixed and Multistage Attention Network for Salient Object Detection'. [paper link](https://arxiv.org/abs/2309.08365)

![](./figures/Overview.png)

## Environment

Python 3.9.13 and Pytorch 1.11.0. Details can be found in `requirements.txt`. 

## Data Preparation
All datasets used can be downloaded at [here](https://pan.baidu.com/s/1fw4uB6W8psX7roBOgbbXyA) [arrr]. 

### Training set
We use the training set of [DUTS](http://saliencydetection.net/duts/) to train our M<sup>3</sup>Net. 

### Testing Set
We use the testing set of [DUTS](http://saliencydetection.net/duts/), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), [PASCAL-S](http://cbi.gatech.edu/salobj/), [DUT-O](http://saliencydetection.net/dut-omron/), and [SOD](https://www.elderlab.yorku.ca/resources/salient-objects-dataset-sod/) to test our M<sup>3</sup>Net. After Downloading, put them into `/datasets` folder.

Your `/datasets` folder should look like this:

````
-- datasets
   |-- DUT-O
   |   |--imgs
   |   |--gt
   |-- DUTS-TR
   |   |--imgs
   |   |--gt
   |-- ECSSD
   |   |--imgs
   |   |--gt
   ...
````

## Training and Testing
1. Download the pretrained backbone weights and put it into `pretrained_model/` folder. [ResNet](https://pan.baidu.com/s/1JBEa06CT4hYh8hR7uuJ_3A) [uxcz], [SwinTransformer](https://github.com/microsoft/Swin-Transformer), [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT), [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) are currently supported. 

2. Run `python train_test.py --train True --test True --record='record.txt'` for training and testing. The predictions will be in `preds/` folder and the training records will be in `record.txt` file. 

## Evaluation
Pre-calculated saliency maps: [M<sup>3</sup>Net-R](https://pan.baidu.com/s/1q4Sp_M-Ph58OsCX1f_c0Ow) [uqsr], [M<sup>3</sup>Net-S](https://pan.baidu.com/s/1m1jF69FaavK4vbPp3B6AcQ) [6jyh]\
Pre-trained weights: [M<sup>3</sup>Net-R](https://pan.baidu.com/s/15vG8N8y-BFv60O_j3C_Uhw) [m789], [M<sup>3</sup>Net-S](https://pan.baidu.com/s/1ZEXR1QD2AMWQfBhxp5f8VA) [4wnw]

For *PR curve* and *F curve*, we use the code provided by this repo: [[BASNet, CVPR-2019]](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool). \
For *MAE*, *Weighted F measure*, *E score* and *S score*, we use the code provided by this repo: [[PySODMetrics]](https://github.com/lartpang/PySODMetrics). 

For more information about evaluation, please refer to `Evaluation/Guidance.md`. 

## Evaluation Results
### Quantitative Evaluation
![](./figures/Quantitative_comparison.png)
![](./figures/Quantitative_comparison2.png)
![](./figures/Quantitative_comparison3.png)

### Precision-recall and F-measure curves
![](./figures/PR&Fm_curves.png)

### Visual Comparison
![](./figures/Visual_comparison.png)

## Acknowledgement
Our idea is inspired by [VST](https://github.com/nnizhang/VST) and [MiNet](https://github.com/lartpang/MINet). Thanks for their excellent works. 
We also appreciate the data loading and enhancement code provided by [plemeri](https://github.com/plemeri), as well as the efficient evaluation tool provided by [lartpang](https://github.com/lartpang/PySODMetrics). 

## Citation
If you think our work is helpful, please cite 
```
@misc{yuan2023m3net,
      title={M$^3$Net: Multilevel, Mixed and Multistage Attention Network for Salient Object Detection}, 
      author={Yao Yuan and Pan Gao and XiaoYang Tan},
      year={2023},
      eprint={2309.08365},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
