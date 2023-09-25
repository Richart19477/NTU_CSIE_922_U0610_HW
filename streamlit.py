import streamlit as st

import cv2
#from PIL import Image
#import imageio
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
   cv2.imwrite("output/1a.png", cv_image_0)

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
   cv2.imwrite("output/1b.png", cv_image_0)

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
   cv2.imwrite("output/1c.png", cv_image_0)

   st.title("Part2. Write a program or use software to do the following requirement.")
   st.subheader("(d) rotate lena.bmp 45 degrees clockwise")

   # cv2.ROTATE_45_CLOCKWISE
   cv_image_0 = cv_image_origin.copy()
   (h, w) = cv_image_0.shape[:2]
   rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), -45, 1.0)   # Note that in cv2 the clockwise is negative
   cv_image_0 = cv2.warpAffine(cv_image_0, rotation_matrix, (w, h))
   st.image(cv_image_0, caption="Rotated Image", use_column_width=False)
   cv2.imwrite("output/2d.png", cv_image_0)
   
   st.subheader("(e) shrink lena.bmp in half")
   st.write("Image shape before shrink: ", cv_image_0.shape)
   cv_image_0 = cv_image_origin.copy()
   (h, w) = cv_image_0.shape[:2]
   cv_image_0 = cv2.resize(cv_image_0, (w//2 ,h//2), interpolation=cv2.INTER_AREA)
   st.image(cv_image_0, caption="Shrinked Image", use_column_width=False)
   st.write("Image shape after shrink: ", cv_image_0.shape)
   cv2.imwrite("output/2e.png", cv_image_0)

   st.subheader("(f) binarize lena.bmp at 128 to get a binary image")
   cv_image_0 = cv_image_origin.copy()
   _, binary_image = cv2.threshold(cv_image_0, 128, 255, cv2.THRESH_BINARY)
   st.image(binary_image, caption="Binary Image", use_column_width=False)
   cv2.imwrite("output/2f.png", cv_image_0)
      

with tab2:
   st.header("HW2")

   st.info("Preload Image Used")
   cv_image_1 = cv2.imread("figs/lena.bmp")

   # Show image   
   st.image(cv_image_1, caption="Original Image", use_column_width=False)

   cv_image_origin = cv_image_1.copy()
   cv_image_1 = cv_image_origin.copy()

   # (a) a binary image (threshold at 128)
   st.subheader("(a) a binary image (threshold at 128)")
   st.write("Image shape: ", cv_image_1.shape)
   for i in range(cv_image_1.shape[0]):
      for j in range(cv_image_1.shape[1]):
         for k in range(cv_image_1.shape[2]):
            if cv_image_1[i][j][k] < 128:
               cv_image_1[i][j][k] = 0
            else:
               cv_image_1[i][j][k] = 255
            
            
   st.image(cv_image_1, caption="a binary image (threshold at 128) Image", use_column_width=False)
   cv2.imwrite("output/2_1a.png", cv_image_1)


   # (b) a histogram
   cv_image_1 = cv_image_origin.copy()

   st.subheader("(b) a histogram")
   # A numpy array with length of 256
   hist_arr = np.zeros(256)

   for i in range(cv_image_1.shape[0]):
      for j in range(cv_image_1.shape[1]):
         for k in range(cv_image_1.shape[2]):
            hist_arr[ cv_image_1[i][j][k] ] += 1
               
   dx = np.arange(256)

   hist_arr //= 3 # The figure supposed to be calculated in B&W

   # Plot the histogram
   # Create Seaborn plot
   plt.figure(figsize=(10, 6))
   sns.barplot(x = dx, y = hist_arr)
   idx = 0
   for label in plt.gca().get_xticklabels():
      if idx % 50 != 0:
         label.set_visible(False)
      idx += 1
   st.pyplot(plt)

   plt.savefig("output/2_1b.png", dpi=None, format='png')

   
   # (c) connected components (regions with + at centroid, bounding box)
   cv_image_1 = cv_image_origin.copy()

   st.subheader("(c) connected components ")
   cv_image_1 = cv2.imread("figs/lena.bmp", cv2.IMREAD_GRAYSCALE) # read image in grayscale

   st.write("Image shape: ", cv_image_1.shape, "(Grayscale)")

   # Transform the image into binary image
   for i in range(cv_image_1.shape[0]):
      for j in range(cv_image_1.shape[1]):
         if cv_image_1[i][j] < 128:
            cv_image_1[i][j] = 0
         else:
            cv_image_1[i][j] = 1

   cv_image_gray = cv_image_1.copy()
   # st.image(cv_image_1, caption="a binary image ", use_column_width=False)

   # Here the 4-connected component policy will be adopted
   # First, we have to create a map with the identical size of original image
   # Second, Check (1) top and (2) left pixel, if they're already 

   ref_matrix = np.zeros(cv_image_1.shape).astype(int)
   idx_now = 1

   # Connected component
   for i in range(cv_image_1.shape[0]):
      for j in range(cv_image_1.shape[1]):
         if cv_image_1[i][j] == 1:
         
            t = ref_matrix[i-1, j] if i - 1 >= 0 else 0
            l = ref_matrix[i, j-1] if j - 1 >= 0 else 0
                
            if t == 0 and l == 0:
               ref_matrix[i, j] = idx_now
               idx_now += 1
            else:
               m = min(t, l)
               ref_matrix[i, j] = m if m > 0 else max(t, l)
               
               if t != l and t > 0 and l > 0:
                  ref_matrix[ ref_matrix == max(t, l) ] = min(t, l)

   #st.write(ref_matrix)

   dict_connected_component = []
   for uniq in np.unique(ref_matrix):
      if uniq == 0:
         continue

      loc = np.where(ref_matrix == uniq)
      loc_list = list(zip(loc[0], loc[1]))
      
      if len( loc_list)  > 500:
         dict_connected_component.append(loc_list)
      
   st.info(f"Number of connected components (>500): {len(dict_connected_component)}")

   
   #cv_image_1 = cv_image_gray.copy()
   cv_image_1 = cv2.imread("figs/lena.bmp", cv2.IMREAD_GRAYSCALE) # read image in grayscale
   for i in range(cv_image_1.shape[0]):
      for j in range(cv_image_1.shape[1]):
         if cv_image_1[i][j] < 128:
            cv_image_1[i][j] = 0
         else:
            cv_image_1[i][j] = 255

   cv_image_1 = np.stack((cv_image_1,)*3, axis=-1)

   for value_list in dict_connected_component:
      value_np = np.asarray(value_list)
      cv_image_1 = cv2.rectangle(cv_image_1, (min(value_np[:, 1]), min(value_np[:, 0])), (max(value_np[:, 1]), max(value_np[:, 0])), (224, 58, 76), 2)
      cv_image_1 = cv2.circle(cv_image_1, (value_np[:,1].mean().astype(int), value_np[:,0].mean().astype(int)) , 2, (242, 139, 50), 5)

      #value_list = [(x2, x1) for x1, x2 in value_list]
      #top_left = min(value_list, key=lambda p: (p[0], p[1]))
      #bottom_right = max(value_list, key=lambda p: (p[0], p[1]))

      #st.write(np.asarray(value_list).shape)
      #st.write("tl & br", top_left, bottom_right)
      #top_left = (min(value[0]), max(value[1]))
      #bottom_right = (max(value[0]), min(value[1]))
      #cv_image_1 = cv2.rectangle(cv_image_1, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]), (224, 58, 76), 2)
      #cv_image_1 = cv2.rectangle(cv_image_1, (min(value_list[1]), min(value_list[0])), (max(value_list[1]), max(value_list[0])), (224, 58, 76), 2)
      
      #cv_image_1 = cv2.rectangle(cv_image_1, top_left, bottom_right, (224, 58, 76), 2)


   st.image(cv_image_1, caption="Connected Component Image", use_column_width=False)