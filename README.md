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
You can download the GRAZPEDWRI-DX Dataset on this [Link](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193).
### Split the dataset
To split the dataset into training set, vvalidation set, and test set, you should first put the image and annotatation into `./GRAZPEDWRI-DX_dataset/data/images`, and `./GRAZPEDWRI-DX_dataset/data/labels`. And then you can split the dataset as the following step:
  ```
    python split.py
  ```

## Methodology
```
  python imgaug.py
```

## Experiments
### Model Training
```
-- model 
-- data 
-- epochs
```

For example, train yolov8n:
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
