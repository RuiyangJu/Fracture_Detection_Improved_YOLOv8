# YOLOv8-AM for Fracture Detection

>[YOLOv8-AM: YOLOv8 with Attention Mechanisms for Pediatric Wrist Fracture Detection](https://arxiv.org/abs/2402.09329)

## :tada::tada::tada: NEWS: [Our split dataset is available now !!!](https://ruiyangju.github.io/GRAZPEDWRI-DX_JU/)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov8-rescbam-yolov8-based-on-an-effective/fracture-detection-on-grazpedwri-dx)](https://paperswithcode.com/sota/fracture-detection-on-grazpedwri-dx?p=yolov8-rescbam-yolov8-based-on-an-effective)

## Architecture
<p align="center">
  <img src="img/figure_architecture.jpg" width="1024" title="details">
</p>

## Performance
| Model | Test Size | Param. | FLOPs | F1 Score | AP<sub>50</sub><sup>val</sup> | AP<sub>50-95</sub><sup>val</sup> | Speed |
| :--: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| YOLOv8 | 1024 | 43.61M | 164.9G | 0.62 | 63.58% | 40.40% | 7.7ms |
| YOLOv8+SA | 1024 | 43.64M | 165.4G | 0.63 | 64.25% | 41.64% | 8.0ms |
| YOLOv8+ECA | 1024 | 43.64M | 165.5G | 0.65 | 64.24% | 41.94% | 7.7ms |
| YOLOv8+GAM | 1024 | 49.29M | 183.5G | 0.65 | 64.26% | 41.00% | 12.7ms |
| YOLOv8+ResGAM | 1024 | 49.29M | 183.5G | 0.64 | 64.98% | 41.75% | 18.1ms |
| YOLOv8+ResCBAM | 1024 | 53.87M | 196.2G | 0.64 | 65.78% | 42.16% | 8.7ms |

## Citation
If you find our paper useful in your research, please consider citing:

**Conference version (accepted by ICONIP 2024)**
```
  @inproceedings{ju2025yolov8,
    title={Yolov8-rescbam: Yolov8 based on an effective attention module for pediatric wrist fracture detection},
    author={Ju, Rui-Yang and Chien, Chun-Tse and Chiang, Jen-Shiun},
    booktitle={International Conference on Neural Information Processing},
    pages={403--416},
    year={2025},
    organization={Springer}
  }
```

**Journal version (accepted by IEEE Access 2025):**
```
  @article{chien2025yolov8,
    title={YOLOv8-AM: YOLOv8 Based on Effective Attention Mechanisms for Pediatric Wrist Fracture Detection},
    author={Chien, Chun-Tse and Ju, Rui-Yang and Chou, Kuang-Yi and Xieerke, Enkaer and Chiang, Jen-Shiun},
    journal={IEEE Access},
    volume={13},
    pages={52461-52477},
    year={2025},
    publisher={IEEE}
  }
```

## Environment
```
  pip install -r requirements.txt
```

## Dataset
* You can find the original GRAZPEDWRI-DX dataset [here](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193) (unsplit).
* Download dataset and put images and annotatation into `./GRAZPEDWRI-DX_dataset/data/images`, `./GRAZPEDWRI-DX_dataset/data/labels`.
  ```
    python split.py
  ```
* The dataset is divided into training, validation, and testing set (70-20-10 %) according to the key `patient_id` stored in `dataset.csv`.
  You can download our split dataset [here](https://1drv.ms/u/s!Ap6uuRvdVcJWbXtfIFYUvzOMKXQ).
* The script then will move the files into the relative folder as it is represented here below.


       GRAZPEDWRI-DX
          └── data   
               ├── meta.yaml
               ├── images
               │    ├── train
               │    │    ├── train_img1.png
               │    │    └── ...
               │    ├── valid
               │    │    ├── valid_img1.png
               │    │    └── ...
               │    └── test
               │         ├── test_img1.png
               │         └── ...
               └── labels
                    ├── train
                    │    ├── train_annotation1.txt
                    │    └── ...
                    ├── valid
                    │    ├── valid_annotation1.txt
                    │    └── ...
                    └── test
                         ├── test_annotation1.txt
                         └── ...


The script will create 3 files: `train_data.csv`, `valid_data.csv`, and `test_data.csv` with the same structure of `dataset.csv`.
                      
### Data Augmentation
* Data augmentation of the training set using the addWeighted function doubles the size of the training set.
```
  python imgaug.py --input_img /path/to/input/train/ --output_img /path/to/output/train/ --input_label /path/to/input/labels/ --output_label /path/to/output/labels/
```
For example:
```
  python imgaug.py --input_img ./GRAZPEDWRI-DX/data/images/train/ --output_img ./GRAZPEDWRI-DX/data/images/train_aug/ --input_label ./GRAZPEDWRI-DX/data/labels/train/ --output_label ./GRAZPEDWRI-DX/data/labels/train_aug/
```
* The path of the processed file is shown below:

       GRAZPEDWRI-DX
          └── data   
               ├── meta.yaml
               ├── images
               │    ├── train
               │    │    ├── train_img1.png
               │    │    └── ...
               │    ├── train_aug
               │    │    ├── train_aug_img1.png
               │    │    └── ...
               │    ├── valid
               │    │    ├── valid_img1.png
               │    │    └── ...
               │    └── test
               │         ├── test_img1.png
               │         └── ...
               └── labels
                    ├── train
                    │    ├── train_annotation1.txt
                    │    └── ...
                    ├── train_aug
                    │    ├── train_aug_annotation1.txt
                    │    └── ...
                    ├── valid
                    │    ├── valid_annotation1.txt
                    │    └── ...
                    └── test
                         ├── test_annotation1.txt
                         └── ...
  
## Methodology
* We have modified the model architecture of YOLOv8 by adding four types of attention modules, including <b>Shuffle Attention (SA), Efficient Channel Attention (ECA), Global Attention Mechanism (GAM), and ResBlock Convolutional Block Attention Module (ResCBAM)</b>.
<p align="center">
  <img src="img/figure_details.jpg" width="1024" title="details">
</p>

## Train & Validate
* We have provided a training set, test set and validation set containing a single image that you can run directly by following the steps in the example below.
* Before training the model, make sure the path to the data in the `./GRAZPEDWRI-DX/data/meta.yaml` file is correct.
```
  # patch: /path/to/GRAZPEDWRI-DX/data
  path: 'E:/GRAZPEDWRI-DX/data'
  train: 'images/train_aug'
  val: 'images/valid'
  test: 'images/test'
```

* Arguments

You can set the value in the `./ultralytics/cfg/default.yaml`.

| Key | Value | Description |
| :---: | :---: | :---: |
| model | None | path to model file, i.e. yolov8m.yaml, yolov8m_ECA.yaml |
| data | None | path to data file, i.e. coco128.yaml, meta.yaml |
| epochs | 100 | number of epochs to train for, i.e. 100, 150 |
| patience | 50 | epochs to wait for no observable improvement for early stopping of training |
| batch | 16 | number of images per batch (-1 for AutoBatch), i.e. 16, 32, 64 |
| imgsz | 640 | size of input images as integer, i.e. 640, 1024 |
| save | True | save train checkpoints and predict results |
| device | 0 | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu |
| workers | 8 | number of worker threads for data loading (per RANK if DDP) |
| pretrained | True | (bool or str) whether to use a pretrained model (bool) or a model to load weights from (str) |
| optimizer | 'auto' | optimizer to use, choices=SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto |
| resume | False | resume training from last checkpoint |
| lr0 | 0.01 | initial learning rate (i.e. SGD=1E-2, Adam=1E-3) |
| momentum | 0.937 | 	SGD momentum/Adam beta1 |
| weight_decay | 0.0005 | optimizer weight decay 5e-4 |
| val | True | validate/test during training |

* Example Train & Val Steps (yolov8m):
```
  python start_train.py --model ./ultralytics/cfg/models/v8/yolov8m.yaml --data_dir ./GRAZPEDWRI-DX/data/meta.yaml
```
* Example Train & Val Steps (yolov8m_ECA):
```
  python start_train.py --model ./ultralytics/cfg/models/v8/yolov8m_ECA.yaml --data_dir ./GRAZPEDWRI-DX/data/meta.yaml
```

## Related Works

<details><summary> <b>Expand</b> </summary>

* [https://github.com/RuiyangJu/Bone_Fracture_Detection_YOLOv8](https://github.com/RuiyangJu/Bone_Fracture_Detection_YOLOv8)
* [https://github.com/RuiyangJu/YOLOv9-Fracture-Detection](https://github.com/RuiyangJu/YOLOv9-Fracture-Detection)
* [https://github.com/RuiyangJu/YOLOv8_Global_Context_Fracture_Detection](https://github.com/RuiyangJu/YOLOv8_Global_Context_Fracture_Detection)
* [https://github.com/RuiyangJu/FCE-YOLOv8](https://github.com/RuiyangJu/FCE-YOLOv8)

</details>
