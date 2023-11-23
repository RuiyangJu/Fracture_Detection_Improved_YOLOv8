## Requirements
* Linux (Ubuntu)
* Python = 3.9
* Pytorch = 1.13.1
* NVIDIA GPU + CUDA CuDNN

## Environment
```
  pip install -r requirements.txt
```

## Dataset
### Download the dataset
* You can download the GRAZPEDWRI-DX Dataset on this [Link](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193).
### Split the dataset
* To split the dataset into training set, vvalidation set, and test set, you should first put the image and annotatation into `./GRAZPEDWRI-DX_dataset/data/images`, and `./GRAZPEDWRI-DX_dataset/data/labels`.
* And then you can split the dataset as the following step:
  ```
    python split.py
  ```

## Methodology
* Data augmentation of the training set using the addWeighted function doubles the size of the training set.
### Data Augmentation
```
  python imgaug.py --input_img /path/to/input/train/ --output_img /path/to/output/train/ --input_label /path/to/input/labels/ --output_label /path/to/output/labels/
```
For example:
```
  python imgaug.py --input_img ./GRAZPEDWRI-DX/data/images/train/ --output_img ./train/ --input_label ./GRAZPEDWRI-DX/data/labels/train/ --output_label ./labels/
```

## Experiments
### Model Training
* Arguments
| Key | Value | Description |
| :---: | :---: | :---: |
| model | None | path to model file, i.e. yolov8n.pt, yolov8n.yaml |
| data | None | path to data file, i.e. coco128.yaml |
| epochs | 100 | number of epochs to train for |
| patience | 50 | epochs to wait for no observable improvement for early stopping of training |
| batch | 16 | number of images per batch (-1 for AutoBatch) |
| imgsz | 640 | size of input images as integer, i.e. 640, 1024 |
| save | True | save train checkpoints and predict results |
| device | None | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu |
| workers | 8 | number of worker threads for data loading (per RANK if DDP) |
| pretrained | True | (bool or str) whether to use a pretrained model (bool) or a model to load weights from (str) |
| optimizer | 'auto' | optimizer to use, choices=SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto |
| resume | False | resume training from last checkpoint |
| lr0 | 0.01 | initial learning rate (i.e. SGD=1E-2, Adam=1E-3) |
| momentum | 0.937 | 	SGD momentum/Adam beta1 |
| weight_decay | 0.0005 | optimizer weight decay 5e-4 |
| val | True | validate/test during training |

* For example, train yolov8n model:
```
  yolo train model=yolov8n.pt data=meta.yaml epochs=100 batch=16 imgsz=640 save=True workers=4 pretrained=yolov8n.pt optimizer=Adam lr0=0.001
```
### Performance Evaluation
<p align="center">
  <img src="img/figure_640.jpg" width="640" title="640">
</p>

<p align="center">
  <img src="img/figure_1024.jpg" width="640" title="1024">
</p>
```
  yolo val model="/path/to/best.pt" data=meta.yaml
```

## Results & Analysis
<p align="center">
  <img src="img/figure_result.jpg" width="640" title="result">
</p>
The prediction examples of our model on the pediatric wrist trauma X-ray images. (a) the manually labeled images, (b) the predicted images.

## Application
### Online Available
You can access our app via the following URL:
```
https://fracture-detection-yolo.streamlit.app/
```
### Run the App on the local
* Use gdown to download the trained model from our GitHub:
```
  gdown https://github.com/RuiyangJu/YOLOv8_CBAM_Fracture_Detection/releases/download/Example_Model/example_model.onnx
```
* You can use our app on your local PC by running it in the following step:
```
  streamlit run application.py
```
