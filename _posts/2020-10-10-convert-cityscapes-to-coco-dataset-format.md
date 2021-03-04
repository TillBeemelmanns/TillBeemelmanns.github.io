---
title: How to convert Cityscapes dataset to CoCo dataset format
tags: Datasets 2D-Object-Detection Segmentation
article_header:
  type: cover
  image:
    src: /assets/images/convert-cityscapes-to-coco/preview.png
---

[Cityscapes](https://www.cityscapes-dataset.com/) is a great dataset for __semantic image segmentation__ which is widely used in academia in the context of automated driving. This dataset provides pixel-precise class annotations on the full image from a vehicle's perspective. However, sometimes you are only interested in the 2D bounding box of specific objects such as `cars` or `pedestrians` in order to perform 2D object detection on the image.

The annotations in Cityscapes also considers segmentation instances. That means a single object is defined by the segmentation mask and an unique instance ID. We can use that information to transform such an instance and extract the extend of it, in short: the __2D bounding box__. Furthermore, we can also determine the area that is covered by that instance, which is called the __mask__. Together, we would obtain labels for __object segmentation__ as shown in the head image above.



### Cityscapes to Coco Conversion Tool
To convert the Cityscapes dataset into a Coco format dataset you may use my [Cityscapes Coco conversion tool](https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion). You can use it as described in the following:


#### Usage
Clone the repository
```bash
git clone https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion
```
and install the requirements:
```
pip install -r requirements.txt 
```
You may setup a [_virtual environment_](https://docs.python.org/3/library/venv.html) to do so.


##### Download the Cityscapes dataset
Download the [Cityscapes dataset](https://www.cityscapes-dataset.com/). Download `gtFine_trainvaltest.zip` and also `leftImg8bit_trainvaltest.zip`. You may have to register in order to download them. Setup the following file structure. 


```
data/
└── cityscapes
    ├── annotations
    ├── gtFine
    │   ├── test
    │   ├── train
    │   └── val
    └── leftImg8bit
        ├── test
        ├── train
        └── val
main.py
inspect_coco.py
README.md
requirements.txt
```

Now you can start the conversion script by calling
```bash
python main.py --dataset cityscapes --datadir="data/cityscapes" --outdir="data/cityscapes/annotations"
```

The script will create the files

* `instancesonly_filtered_gtFine_train.json`
* `instancesonly_filtered_gtFine_val.json`

in the directory `annotations` for the `train` and `val` split which contain the Coco annotations.


#### Filter certain classes
The Cityscapes dataset contains about 30 different classes. Not all of them may be relevant for you. The variable `category_instancesonly` defines which classes should be considered in the conversion process. 


```python
category_instancesonly = [
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
]
```

On these mentioned classes will be converted into a Coco annotation. 


Sometimes the segmentation annotations are so small that no reasonable big enough object could be created. In this case the, the object will be skipped.
```
Warning: invalid contours.
```

### Output
You can visualize the final __object segmentation__ annotations with the inspection script:

```
python inspect_coco.py --coco_dir data/cityscapes
```
And you would obtain some of the following pictures:

![](/assets/images/convert-cityscapes-to-coco/plot1.png)
![](/assets/images/convert-cityscapes-to-coco/plot2.png)



#### Wrap-up
- You converted a __image segmentation__ dataset into a __object segmentation__ dataset
- You can now use the new dataset with [Mask R-CNN](https://github.com/matterport/Mask_RCNN), [DETR](https://github.com/facebookresearch/detr) or [Detectron2](https://github.com/facebookresearch/detectron2) network architectures
