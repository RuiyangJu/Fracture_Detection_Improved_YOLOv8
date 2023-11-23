import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import onnxruntime as ort
from matplotlib.colors import TABLEAU_COLORS 

ALLOWED_EXTENSIONS = {"txt", "pdf", "bmp", "png", "jpg", "jpeg", "gif"}
h, w = 640, 640
model_path = os.path.join("./", "example_model.onnx")
device = "cuda"

def color_list():
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    return [hex2rgb(h) for h in TABLEAU_COLORS.values()]
colors = color_list()

def xy2whn(bbox, H, W):
    x1, y1, x2, y2 = bbox
    return [0.5*(x1+x2)/W, 0.5*(y1+y2)/H, (x2-x1)/W, (y2-y1)/H]

def whn2xy(bbox, H, W):
    x, y, w, h = bbox
    return [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]

def load_img(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image[..., ::-1]

def preprocess(img):
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32).transpose(2, 0, 1)/255
    return np.expand_dims(img, axis=0)

def model_inference(model_path, image_np, device="cpu"):
    providers = ["CUDAExecutionProvider"] if device=="cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image_np})
    return output[0][:, :6]

def post_process(img, output, score_threshold=0.3):
    det_bboxes, det_scores, det_labels = output[:, 0:4], output[:, 4], output[:, 5]
    id2names = {
        0: "boneanomaly", 1: "bonelesion", 2: "foreignbody", 
        3: "fracture", 4: "metal", 5: "periostealreaction", 
        6: "pronatorsign", 7:"softtissue", 8:"text"
    }

    if isinstance(img, str):
        img = cv2.imread(img)
    
    img = img.astype(np.uint8)
    H, W = img.shape[:2]
    label_txt = ""

    for idx in range(len(det_bboxes)):
        if det_scores[idx]>score_threshold:
            bbox = det_bboxes[idx]
            label = det_labels[idx]
            bbox = xy2whn(bbox, h, w)
            label_txt += f"{int(label)} {det_scores[idx]:.5f} {bbox[0]:.5f} {bbox[1]:.5f} {bbox[2]:.5f} {bbox[3]:.5f}\n"
            bbox = whn2xy(bbox, H, W)
            bbox_int = [int(x) for x in bbox]
            x1, y1, x2, y2 = bbox_int
            color_map = colors[int(label)]
            txt = f"{id2names[label]} {det_scores[idx]:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color_map, 2)
            cv2.rectangle(img, (x1-2, y1-text_height-10), (x1 + text_width+2, y1), color_map, -1)
            cv2.putText(img, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img, label_txt

if __name__ == "__main__":
    # st.title('Fracture Detection on X-ray Images')

    image = Image.open('logo.jpg')
    st.image(image)

    st.subheader('You can use the example image we provided for testing if you do not have an x-ray image of the wrist injury:', divider='rainbow')
    with open("example_1.png", "rb") as file_1, open("example_2.png", "rb") as file_2, open("example_3.png", "rb") as file_3:
        btn_1 = st.download_button(
                label="Download Example-1 Image",
                data=file_1,
                file_name="example_1.png",
                mime="image/png"
        )

        btn_2 = st.download_button(
                label="Download Example-2 Image",
                data=file_2,
                file_name="example_2.png",
                mime="image/png"
            )

        btn_3 = st.download_button(
                label="Download Example-3 Image",
                data=file_3,
                file_name="example_3.png",
                mime="image/png"
            )    
    
    st.subheader('Please upload the x-ray image before you perform fracture detection:', divider='rainbow')
    uploaded_file = st.file_uploader("Upload the image file", type=["png", "bmp", "jpg", "jpeg", "gif"])

    if uploaded_file is not None:
        conf_thres = st.slider("Object Confidence Threshold", 0.2, 1., step=0.05)
        img = load_img(uploaded_file)
        img_preprocess = preprocess(img)
        out = model_inference(model_path, img_preprocess, device)
        out_img, out_txt = post_process(img, out, conf_thres)
        st.image(out_img, caption="Prediction", channels="RGB")
        col1, col2 = st.columns(2)
        col1.download_button(
            label="Download Prediction",
            data=cv2.imencode(".png", out_img[..., ::-1])[1].tobytes(),
            file_name=uploaded_file.name,
            mime="image/png"
        )
        col2.download_button(
            label="Download Detection",
            data=out_txt,
            file_name=uploaded_file.name[:-4] + ".txt",
            mime="text/plain"
        )

    st.subheader('Please email us when you have any problems using this application:', divider='rainbow')
    st.caption('Mr. RuiYang :email: : jryjry1094791442@gmail.com')
