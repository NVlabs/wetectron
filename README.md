# Wetectron

Wetectron is a software system that implements state-of-the-art weakly-supervised object detection algorithms.

![Wetectron](docs/teaser.png) 

### Project [CVPR'20](https://jason718.github.io/project/wsod/main.html), [ECCV'20](https://jason718.github.io/project/ufoo/main.html) | Paper [CVPR'20](https://arxiv.org/pdf/2004.04725.pdf), [ECCV'20](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640290.pdf) 

## Installation

Check [INSTALL.md](docs/INSTALL.md) for installation instructions.

## Partial labels

The simulated partial labels (points and scribbles) of COCO can be found at [Google-drive](https://drive.google.com/drive/u/2/folders/1DYKIOrM0X3o_kdA-p932XYcIzku2fKAM) or [Dropbox](https://www.dropbox.com/sh/tq2gasoik98sfyc/AADdK9zPne9_v2QkauO2kpTZa?dl=0). 

Please check `tools/vis_partial_labels.ipynb` for a visualization example.

## Model zoo
Check [MODEL_ZOO.md](docs/MODEL_ZOO.md) for detailed instructions.

## Getting started

Check [GETTING_STARTED](docs/GETTING_STARTED.md) for detailed instrunctions. 

## New dataset
If you want to run on your own dataset or use other pre-computed proposals (e.g., Edge Boxes), please check [USE_YOUR_OWN_DATA](docs/USE_YOUR_OWN_DATA.md) for some tips.

## Misc

Please also check the documentation of [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) for things like abstractions and troubleshooting.
If your issues are not present there, feel free to open a new issue.

## Todo:
1. Sequential back-prop and ResNet models.

## Citations

Please consider citing following papers in your publications if they help your research. 

```BibTeX
@inproceedings{ren-cvpr020,
  title = {Instance-aware, Context-focused, and Memory-efficient Weakly Supervised Object Detection},
  author = {Zhongzheng Ren and Zhiding Yu and Xiaodong Yang and Ming-Yu Liu and Yong Jae Lee and Alexander G. Schwing and Jan Kautz},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}

@inproceedings{ren-eccv2020,
  title = {UFO$^2$: A Unified Framework towards Omni-supervised Object Detection},
  author = {Zhongzheng Ren and Zhiding Yu and Xiaodong Yang and Ming-Yu Liu and Alexander G. Schwing and Jan Kautz},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020}
}
```

## License

This code is released under the [Nvidia Source Code License](LICENSE). 

This project is built upon [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), which is released under [MIT License](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/LICENSE).
