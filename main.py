import streamlit as st
import os
import torch
from torchvision import models , transforms
import torch.nn as nn
from PIL import Image

device = 'cpu'
transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4,0.4,0.4],[0.2,0.2,0.2])
])

model_ft = models.resnet18(weights='IMAGENET1K_V1')
for params in model_ft.parameters():
    params.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load('D:/current_work/Pytorch_image_classification/model.pth',map_location=device))
classes = ['not_pizza', 'pizza']


st.title('Pizza Lover')
st.markdown('''![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)```3.9```
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

![alt text](https://storage.googleapis.com/kaggle-datasets-images/2296957/3863171/7be054fab196beabd5b0b4a462c31c21/dataset-cover.jpg?t=2022-06-26-02-08-07)''')

st.subheader('Predict weather an image is of pizza or not')
uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'png'])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open(os.path.join(uploaded_file.name),"wb") as f: 
      f.write(uploaded_file.getbuffer())
    img = Image.open(uploaded_file.name).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    model_ft.eval()
    with torch.no_grad():
        output = model_ft(img)
        _ , pred = torch.max(output,1)
        ans = classes[pred[0]]
    st.subheader(ans)
    st.image(bytes_data)
    os.remove(uploaded_file.name)

st.write('made with love :heart: by lakshit karsoliya github : https://github.com/Lakshit-Karsoliya')

