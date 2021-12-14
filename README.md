# pytorch-segmentation-pipeline
Implementation of UNet &amp; DeepLabV3+ are from different source.

For purpose of my workflow simplification, 
implementation of model training and inference pipeline.
Segmentation models are U-Net[1] and DeepLabV3+[2].

-Each experiments separeted in directory. (I.e., default exp0)
-Experiment results are in one directory: model weights, logs, infernce results are stored in the experiment directory.
-Inferences on images and videos are done in similar manner, no need to point out. Also in pointed experiment.

For execution command, check main.py's arguments.
Example data directory is included.


-------------------------------------------------------------------------------------------------------------
Model U-Net & DeepLabV3+ architectures are from source in the below.
U-Net architecture's source: nikhilroxtomar
DeepLabV3+ architecture's source: jfzhang95 

[1]. Olaf Ronneberger, Philipp Fischer, Thomas Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", (2015), arXiv:1505.04597 [cs.CV].
https://arxiv.org/pdf/1505.04597.pdf

[2]. Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam, "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation", (2018), arXiv:1802.02611 [cs.CV].
https://arxiv.org/pdf/1802.02611v3.pdf
