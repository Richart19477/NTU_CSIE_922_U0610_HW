import streamlit as st

import cv2
#from PIL import Image
#import imageio
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns


# Page config
st.set_page_config(
    page_title="NTU CV 2023 - r12922a08",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Stable Variances:
if 'state_1' not in st.session_state:
    st.session_state['state_1'] = 0

# Page title
st.title('NTU CSIE_5732 : Computer Vision 2023')
st.subheader("電腦視覺Homework")

# Page announcement
#st.caption("Github: [Github link](https://github.com/RoyChao19477/)")
st.caption("Author: R12922A08@2023")

# Tabe
tab1, tab2 = st.tabs(["HW1", "HW2"])
with tab1:
   st.title("HW1")
   st.header("Part1. Write a program to do the following requirement.")

   # button
   image_0 = None
   cv_image_0 = None
   if st.button("Upload Image by yourself here"):
      st.session_state.state_1 = 1
   if st.session_state.state_1 >= 1:
      # Upload Image
      image_0 = st.file_uploader("Upload Image", type=['jpg', 'png', 'bmp'])
      if image_0 is not None:
         file_bytes = np.asarray(bytearray(image_0.read()), dtype=np.uint8)
         cv_image_0 = cv2.imdecode(file_bytes, 1)
         st.write(cv_image_0.shape)
         # Fix color issue
         cv_image_0[:,:, [0, 2]] = cv_image_0[:,:, [2, 0]]
         st.info("Successfully uploaded!")
   else: #if image_0 is None:
      st.info("Preload Image Used")
      cv_image_0 = cv2.imread("figs/lena.bmp")

   # Show image   
   st.image(cv_image_0, caption="Original Image", use_column_width=False)

   cv_image_origin = cv_image_0.copy()
   cv_image_0 = cv_image_origin.copy()


   # 1.a
   st.subheader("(a) upside-down lena.bmp")
   st.write("Image shape: ", cv_image_0.shape)
   for i in range(cv_image_0.shape[0] // 2):
      for j in range(cv_image_0.shape[1]):
         for k in range(cv_image_0.shape[2]):
            tmp = cv_image_0[i][j][k]
            
            cv_image_0[i][j][k] = cv_image_0[cv_image_0.shape[0] - i - 1][j][k]
            cv_image_0[cv_image_0.shape[0] - i - 1][j][k] = tmp
   st.image(cv_image_0, caption="Upside-down Image", use_column_width=False)


   # 1.b
   st.subheader("(b) right-side-left lena.bmp")
   cv_image_0 = cv_image_origin.copy()
   for i in range(cv_image_0.shape[0]):
      for j in range(cv_image_0.shape[1] // 2):
         for k in range(cv_image_0.shape[2]):
            tmp = cv_image_0[i][j][k]
            
            cv_image_0[i][j][k] = cv_image_0[i][cv_image_0.shape[1] - j - 1][k]
            cv_image_0[i][cv_image_0.shape[1] - j - 1][k] = tmp

   st.image(cv_image_0, caption="Upside-down Image", use_column_width=False)
   
   st.subheader("(c) diagonally flip lena.bmp")

   # 1.c
   cv_image_0 = cv_image_origin.copy()
   for i in range(cv_image_0.shape[0]):
      for j in range(i):
         for k in range(cv_image_0.shape[2]):
            tmp = cv_image_0[i][j][k]
            
            cv_image_0[i][j][k] = cv_image_0[j][i][k]
            cv_image_0[j][i][k] = tmp

   st.image(cv_image_0, caption="Diagonally-flipped Image", use_column_width=False)

   st.title("Part2. Write a program or use software to do the following requirement.")
   st.subheader("(d) rotate lena.bmp 45 degrees clockwise")

   # cv2.ROTATE_45_CLOCKWISE
   cv_image_0 = cv_image_origin.copy()
   (h, w) = cv_image_0.shape[:2]
   rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), -45, 1.0)   # Note that in cv2 the clockwise is negative
   cv_image_0 = cv2.warpAffine(cv_image_0, rotation_matrix, (w, h))
   st.image(cv_image_0, caption="Rotated Image", use_column_width=False)
   
   st.subheader("(e) shrink lena.bmp in half")
   st.write("Image shape before shrink: ", cv_image_0.shape)
   cv_image_0 = cv_image_origin.copy()
   (h, w) = cv_image_0.shape[:2]
   cv_image_0 = cv2.resize(cv_image_0, (w//2 ,h//2), interpolation=cv2.INTER_AREA)
   st.image(cv_image_0, caption="Shrinked Image", use_column_width=False)
   st.write("Image shape after shrink: ", cv_image_0.shape)

   st.subheader("(f) binarize lena.bmp at 128 to get a binary image")
   cv_image_0 = cv_image_origin.copy()
   _, binary_image = cv2.threshold(cv_image_0, 128, 255, cv2.THRESH_BINARY)
   st.image(binary_image, caption="Binary Image", use_column_width=False)
      

with tab2:
   st.header("HW2")
   st.warning("Coming soon ...")
