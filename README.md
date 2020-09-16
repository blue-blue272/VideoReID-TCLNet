# TCLNet
This repository contains the code for the paper:
<br>
[**Temporal Complementary Learning for Video Person Re-Identification**](https://arxiv.org/pdf/2007.09357.pdf)
<br>
Ruibing Hou, Hong Chang, Bingpeng Ma, Shiguang Shan, Xilin Chen
<br>
ECCV 2020


### Abstract

This paper proposes a Temporal Complementary Learning Network that extracts complementary features of consecutive video frames for video person re-identification. Firstly, we introduce a Temporal Saliency Erasing (TSE) module including a saliency erasing operation and a series of ordered learners. Specifically, for a specific frame of a video, the saliency erasing operation drives the specific learner to mine new and complementary parts by erasing the parts activated by previous frames. Such that the diverse visual features can be discovered for consecutive frames and finally form an integral characteristic of the target identity. Furthermore, a Temporal Saliency Boosting (TSB) module is designed to propagate the salient information among video frames to enhance the salient feature. It is complementary to TSE by effectively alleviating the information loss caused by the erasing operation of TSE. Extensive experiments show our method performs favorably against stateof-the-arts. 

### Training and test

  ```Shell
  # For MARS
  python train.py --root your path to MARS --save_dir log-mars-tclnet #
  python test.py --root your path to MARS 
  
  ```

### Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{TCLNet,
  title={Temporal Complementary Learning for Video Person Re-Identification},
  author={Ruibing Hou and Hong Chang and Bingpeng Ma and Shiguang Shan and Xilin Chen},
  booktitle={ECCV},
  year={2020}
}
```

### Platform
This code was developed and tested with pytorch version 1.0.1.


## Acknowledgments

This code is based on the implementations of [**Deep person reID**](https://github.com/KaiyangZhou/deep-person-reid/tree/master/torchreid).
