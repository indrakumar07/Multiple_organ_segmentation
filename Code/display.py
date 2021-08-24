import streamlit as st
import brain
import numpy as np
import cv2
import os
import base64
import nibabel as nib


def disp_png_jpg(img,dataset):
    img=img.getvalue()
    nparr=np.fromstring(img,np.uint8)
    im =  cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    temp=im
    test_img = cv2.resize(im, (256,256))
    test_img = cv2.cvtColor(test_img, cv2.IMREAD_COLOR)
    test_img = np.expand_dims(test_img, axis=0)
    cols = st.columns(2)
    cols[0].image(test_img,caption="Input Image",width=256)
    x=cols[1].image('Images/load.gif')
    pre,y,himg=brain.predict(test_img,dataset,temp)
    if y==0:
        x.empty()
        cols[1].image('Images/notumor.gif')
    else:
        x.empty()
        st.image(himg,caption="Heatmap Image",use_column_width=True)
        cols[1].image(pre,caption="Segmented Mask",width=256)        
       
        
def get_binary_file_downloader_html(bin_file, file_label='File'):
        bin_str = base64.b64encode(bin_file).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        st.markdown(get_binary_file_downloader_html("x", 'Segmented Liver'), unsafe_allow_html=True)
        return href
    

def disp_nii(img,store): 
    place=st.image('Images/processing.gif')
    x=brain.predict_nii(img,place)    
    if x==0:
        st.write("Exception occured :FileNotFound")
        place.empty()
        place.image('Images/error.gif')
        
    else:
        try:
            nib.save(x, store+'/segmented_mask.nii')
            place.empty()
            place.image('Images/completed.gif')
        except:
            place.empty()
            place.image('Images/error.gif')

    
