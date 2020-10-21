# Run on your own dataset
If you want to test weakly-supervised object detection on your own data, we provide some tips to ease the process.

# Datasets
You can also configure your own paths to the datasets.
For that, all you need to do is to modify `wetectron/config/paths_catalog.py` to
point to the location where your dataset is stored.
You can also create a new `paths_catalog.py` file which implements the same two classes,
and pass it as a config argument `PATHS_CATALOG` during training.

# Proposals
You can use alternative methods such as Edge Boxes for computing proposals. To create a proposal file, you need to save a pickle file with two keys `boxes` and `indexes`. 
Note that the `indexes` have to be consistent with your dataset.
Please check the provided proposal files for details. Use `encoding="latin1` as loading flag:
```
with open('MCG-coco_2014_minival-boxes.pkl', 'rb') as f:
    proposals = pickle.load(f, encoding="latin1")
```

# Configurations
In your configuration files, set `MODEL.ROI_BOX_HEAD.NUM_CLASSES` to the number of classes of your dataset. 
Change `DATASETS` and `PROPOSAL_FILES` to the corresponding files generated from the above steps.
Tunning the hyperparameters in `SOLVER.*` based on your validation results.

Feel free to post the issues you meet during the process!
