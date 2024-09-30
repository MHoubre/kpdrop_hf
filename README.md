# Implementation for KDRop using HuggingFace trainer

This repository is a reimplementation of the KPDrop-A augmentation method from [KPDrop: Improving absent keyphrase generation](https://aclanthology.org/2022.findings-emnlp.357)(Chowdhury et al, 2022)
The only difference in the augmentation process compared to the original method is that our implementation does the augmentation on the entire dataset before using it for training. The implementation from [JRC1995 KProp](https://github.com/JRC1995/KPDrop/tree/main) does the augmentation during batch collection and augments the batch itself.
This difference in implementation makes the model see the origin instance and the new modified instance in the same batch whereas our implementation makes it a little bit harder as the final dataset is shuffled.
