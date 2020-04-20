# InstanceSeg-Without-Voxelwise-Labeling
[Instance Segmentation from Volumetric Biomedical Images Without Voxel-Wise Labeling](https://link_springer.gg363.site/chapter/10.1007/978-3-030-32245-8_10)

This code is based on Mask RCNN in pytorch version from "https://github.com/roytseng-tw/Detectron.pytorch", and the main improvements are as below:  
1) Reimplement corresponding operation modules in 3D style for volumetric data.  
2) Merge PRM into the framework under Faster RCNN mode. ("https://github.com/ZhouYanzhao/PRM")  
3) Add an effecient binarization function based on 2D Otsu algorithm, which can help finish instance segmentation based on detection and PRM results.   
4) Add some useful and efficient evaluation code for detection and instance segmentation in 3D format for volumetric data.  

Prerequisites  

1) refer to https://github.com/roytseng-tw/Detectron.pytorch  
2) refer to https://github.com/ZhouYanzhao/PRM  
3) other:  
         scikit-image  
         libtiff  
4) the code is tested on python3.6 and pytorch0.4.0  

Usage  

1) Data  
   soma: the data contains 3d-tif images and txt soma spots labels, only a small amount of test images have instance mask labels in 3d-tif format.  
   nuclei: the data contains 3d-tif images and 3d-tif instance mask labels.  
2) Directory  
   a) "lib/datasets/voc_dataset_evaluator.py" set your dataset path.  
   b) "configs/soma_starting/e2e_mask_rcnn_soma_dsn_body.yaml" and "configs/cell_tracking_baseline/e2e_mask_rcnn_N3DH_SIM_dsn_body.yaml" set your catch path (DATA_DIR) for temperary roidb.  
   c) "train.sh" set your python exe path.  
3) Training  
   Run "tools/train_net_step.py" with setting the DEBUG as True or run "train.sh" with DEBUG as False, note that the former will not save snapshort models.  
4) Testing  
   a) Well-trained models are "model/nuclei/model_step8999.pth" and "model/soma/model_step5999.pth" for two datasets.  
   b) Config files are "configs/cell_tracking_baseline/e2e_mask_rcnn_N3DH_SIM_dsn_body.yaml" and "configs/soma_starting/e2e_mask_rcnn_soma_dsn_body.yaml" for two datasets.  
   c) Set PRM_ON as True in "***.yaml" if generating corresponding PRM results is needed.  
   d) Run "tools/infer_simple.py" and with proper modifications to args.  
5) Binarization  
   a) "tools/binarization_nuclei.py" for nuclei data binarization.  
   b) "tools/binarization_soma.py" for soma data binarization.  
6) Evaluation  
   a) "tools/evaluation/eval_instance_segmentation_soma.py" for evaluating soma instance segmentation performance, and the metric is AP.  
   b) "tools/evaluation/eval_instance_segmentation_soma_ngps.py" for evaluating soma instance segmentation performance of NeuroGPS or semantic segmentation method like DSN, and the metric is AP.  
   c) "tools/evaluation/evaluation_nuclei_f1score.py" for evaluating nulei detection performance and the metric is F1-score.   
   d) "tools/evaluation/evaluation_nuclei_f1score_seg.py" for evaluating nulei instance segmentation performance and the metric is F1-score.  

Citation

If you find the code useful for your research, please cite: 
Dong M, Liu D, Xiong Z, et al. Instance Segmentation from Volumetric Biomedical Images Without Voxel-Wise Labeling[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2019: 83-91.

Note:
There is a typo in the experiment part of our MICCAI paper： The input size for our mouse brain dataset is 128*256*256 instead of 12*256*256.
