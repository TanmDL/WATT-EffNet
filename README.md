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
