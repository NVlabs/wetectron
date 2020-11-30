## Model zoo

The pretrained models can be found at [Google-drive](https://drive.google.com/drive/u/2/folders/1DYKIOrM0X3o_kdA-p932XYcIzku2fKAM) or [Dropbox](https://www.dropbox.com/sh/gb2y2j5cmry2ire/AABj7P6fV1pyxcwAdKp-gnNWa?dl=0). 

Note that all these models were trained using 8 NVIDIA V100 GPUs (16/32 GB) and PyTorch 1.0. 
The current code has been upgraded to support latest version of PyTorch.

## coco models

| Train data              | Eval data        | Config                       | Backbone     | AP    | AP50   |
|:------------------------|------------------|------------------------------|--------------|-------|-------:|
| train+valminusminival   | minival          | coco/V_16_coco17.yaml        | VGG-16       | 13.0  | 25.8   |
| train                   | val              | coco/V_16_coco14.yaml        | VGG-16       | 11.7  | 24.0   |

We used coco 2014 naming rule, `coco_2017_train` = `train` + `valminusminival` , `COCO_2017_val` = `minival`.
Note that these two models are trained without using concrete Dropblock to save training time. 
The released models have acheived slightly better results (AP) than the numbers in CVPR'20 paper Tab.1 & 4.

### partial labels

| Train data              | Eval data        | Config                       | Backbone     | AP    | AP50   |
|:------------------------|------------------|------------------------------|--------------|-------|-------:|
| train                   | val              | coco/V_16_coco17_point.yaml  | VGG-16       | 12.7  | 27.4   |
| train                   | val             | coco/V_16_coco14_scribble.yaml| VGG-16       | 14.3  | 30.7   |

These two models are trained using partial labels. 
The released models have acheived slightly better results than the numbers in ECCV'20 paper Tab.2.

## voc models

| Train data        | Eval data        | Config                       | Backbone     | mAP    |
|:------------------|------------------|------------------------------|--------------|-------:|
| voc 2007          | voc 2007 test    | voc/V_16_voc07.yaml          | VGG-16       | 54.9   |
| voc 2007+2012     | voc 2007 test    | voc/V_16_voc0712.yaml        | VGG-16       | 58.1   |

The results are reported in the CVPR'20 paper Tab.3. Note that `voc 2007 test` is publically available and thus should be treated as a validation set.
People usually tune the hyper-parameters on this split and report the best number in paper.
For a fair comparison, please use voc 2012 split and use the private test set. 
You can find our results (last row in the table) [here](http://host.robots.ox.ac.uk:8080/anonymous/DCJ5GA.html).
Since `voc` dataset is relatively small in all the splits, the model is easy to overfit and you might see some variance on the val set. 
Please read the 2nd issue in [GETTING_STARTED](docs/GETTING_STARTED.md) `known_issues` for some tips.
