## Dataset

Download the COCO/VOC dataset, and then

```bash
# symlink the coco dataset
mkdir -p datasets/coco
ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014

# for pascal voc dataset:
mkdir -p datasets/voc
ln -s /path_to_VOCdevkit/VOC2007 datasets/voc/VOC2007
ln -s /path_to_VOCdevkit/VOC2012 datasets/voc/VOC2012
```

P.S. `COCO_2017_train` = `COCO_2014_train` + `valminusminival` , `COCO_2017_val` = `minival`

## Proposals

Download the proposals from [Google-drive](https://drive.google.com/drive/u/2/folders/1DYKIOrM0X3o_kdA-p932XYcIzku2fKAM) or [Dropbox](https://www.dropbox.com/sh/u7txwf3l084k0l9/AAB_PiIP33D_UgYi8AFUzRQ3a?dl=0).

```bash
# We provide MCG proposals for COCO, and Selective-Search (SS) proposals for PASCAL VOC. 
# by default
mkdir proposal
ln -s  /path/to/downloaded/files/*.pkl proposal/
# You may also change config files to set proposal path
```

## Evaluation

Here is an example to evaluate the released model:
```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
    --config-file "configs/voc/V_16_voc07.yaml" TEST.IMS_PER_BATCH 8 \
    OUTPUT_DIR /path/to/output/dir \
    MODEL.WEIGHT /path/to/model
```
Example results will be dumped in the `/path/to/output/dir` folder. You can use flag `--vis` to generate some visualizations. 

## Training

All the configuration files that we provide assume using 8 GPUs. 
```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
    --config-file "configs/voc/V_16_voc07.yaml" --use-tensorboard \
    OUTPUT_DIR /path/to/output/dir
```

### Known issues
1. If you get error `RuntimeError: CUDA error: device-side assert triggered`, check issue #22, or simply re-lanunch again.
2. Since the voc datasets are very small, the best results usually are not achieved in the last epoch.
Please save the intermediate models frequently (change `SOLVER.CHECKPOINT_PERIOD`) and check validation (`voc 2007 test`) results 
(especially these models around the 1st learning rate dropping time). 
You can also change the random seed `SEED` to a different value or `-1` (random).
Note that setting `SEED` to a fixed value still cannot guarantee deterministic behavior, see explanations 
[here](https://github.com/facebookresearch/detectron2/blob/e0e166d864a2021a15a2bc2c9234d04938066265/detectron2/config/defaults.py#L604).
3. `OSError: Cannot allocate memory`: please check [this thread](https://github.com/prlz77/ResNeXt.pytorch/issues/5)
4. The displayed `max mem` doesn't align with the trun memory usage. 
