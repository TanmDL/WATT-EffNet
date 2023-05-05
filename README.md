# WATT-EffNet
WATT-EffNet: A Lightweight and Accurate Model for Classifying Aerial Disaster Images

Authors: Gao Yu Lee, Tanmoy Dam, Md Meftahul Ferdaus, Daniel Puiu Poenar and Vu N. Duong

Abstract: Incorporating deep learning (DL) classification models into unmanned aerial vehicles (UAVs) can significantly aug
ment search-and-rescue operations and disaster management efforts. In such critical situations, the UAV’s ability to promptly
comprehend the crisis and optimally utilize its limited power and processing resources to narrow down search areas is crucial.
Therefore, developing an efficient and lightweight method for scene classification is of utmost importance. However, current
approaches tend to prioritize accuracy on benchmark datasets at the expense of computational efficiency. To address this 
shortcoming, we introduce the Wider ATTENTION EfficientNet (WATT-EffNet), a novel method that achieves higher accuracy 
with a more lightweight architecture compared to the baseline EfficientNet. The WATT-EffNet leverages width-wise incremental
feature modules and attention mechanisms over width-wise features to ensure the network structure remains lightweight.
We evaluate our method on a UAV-based aerial disaster image classification dataset and demonstrate that it outperforms the
baseline by up to 15 times in terms of classification accuracy and 38.3% in terms of computing efficiency as measured by Floating
Point Operations per second (FLOPs). Additionally, we conduct an ablation study to investigate the effect of varying the width
of WATT-EffNet on accuracy and computational efficiency.

See WATT-EffNet Structure FINAL.png for an illustration of our algorithmic architecture design.

(This github repository is still updating in progress. Stay tune for more information.)

# Dataset

Our work is trained and evaluated on the Aerial Image Database for Emergency Response (AIDER) subset by Kyrkou and Theocharides [1]. The dataset comprised of images illustrating four major types of disasters: fire, floods, afermath of building collapses and traffic collisions, as well as images of non-disasters (normal class) in a relatively larger amount than the other four to replicate real-world scenario as close as possible. Some samples of the images of each class is shown in AIDER Images Examples.png. Unlike the original dataset which comprised of a total of 8540 images, the subset only contained 6433 images. The AIDER subset can be downloaded from https://zenodo.org/record/3888300.

AIDER subset image sets distribution in our approach:

| Class | Train | Valid | Test | Total per Class |
| ------ | ------| ------| ------| ------|
|**Collapsed Building**| 367 | 41 | 103 | 511 |
|**Fire**| 249 | 63 | 209 | 521 |
|**Flood**| 252 | 63 | 211 | 526 |
|**Traffic**| 232 | 59 | 194 | 485 |
|**Normal**| 2107 | 527 | 1756 | 4390 |
|**Total per Set**| 3207 | 753 | 2473 | **6433** |

