# Guidance for the evaluation of SOD results
Most of the previous SOD methods' prediction maps can be found in [VST](https://github.com/nnizhang/VST) [SOTA Saliency Maps for Comparison], so you don't need to search for them one by one. 
## *MAE*, *Weighted F measure*, *E score* and *S score*
We recommend you to use the code provided by this repo: [[PySODMetrics]](https://github.com/lartpang/PySODMetrics) for SOD quantitative evaluation. This repo covers all the evaluation indicators involved in previous methods and has high operational efficiency. In addition, consistency in the evaluation results is also important, and we hope that sod related work can use the same repo to evaluate the model's prediction results 

After downloading this repo you need to modify `dataset.json` and `method.json` yourself, and then you can call `eval.py` to perform the evaluation. The `README.md` of this repo provides detailed guidance. 

## *PR curve* and *F curve*
Although the repo mentioned above also provides the generation of pr or fm curves, we do not recommend using it because the intuition and aesthetics are more important for curve graphs. we use the code provided by this repo: [[BASNet, CVPR-2019]](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool) to draw curve graphs,  while the code for this repo can be slightly improved: 
1. In previous methods, inconsistent testing datasets were used when generating prediction graphs, or generated prediction graphs containing more than one channel, which could cause interruption in the running of the profiling code. We have made some modifications at `measures.py` so that the code can ignore these extreme prediction graphs, which is negligible for the final evaluation results. 
2. When drawing curves, we do not need to perform another evaluation on all the predicted plots of the methods, so Jupyter Notebook (e.g., `demo.ipynb`) may be a more suitable way for us to adjust the parameters of the plot more conveniently. 

Besides, you need to prepare predictions of all evaluation methods and arrange the file structure as follows: 
```
-- eval
    |--M3Net
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
    |--VST
        |-- DUT-O
        ...
    ...
```