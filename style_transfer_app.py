import altair as alt
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import PIL as Image


hub_handle = 'https://tfhub.dev/google/arbitrary-image-stylization-v1-256/2'

hub_module = hub.load(hub_handle)

def crop_image(img):
    curr_shape  = img.shape
    new_shape = min(curr_shape[0], curr_shape[1])
    offset_y = max(curr_shape[0] - curr_shape[1],0)
    offset_x = max(curr_shape[1] - curr_shape[0],0)
    img = tf.image.crop_to_bounding_box(img,offset_y,offset_x,new_shape)
    return img


def load_image(upload_file, image_size = (256,256), col = st):
    img = Image.open(upload_file)
    img = tf.convert_to_tensor(img)
    img = crop_image(img)
    img = tf.image.resize(img, image_size)
    if img.shape[-1] == 4:
        img = img[:,:,:3]
    
    img = tf.reshape(img,[-1,image_size[0],image_size[1],3])/255
    col.image(np.array(img))


def show_image(images, title = ('',), col = st):
    n = len(images)
    for i in range (n):
        col.image(np.array(images[i][0]))

st.set_page_config(layout = 'wide')

alt.renderers.set_embed_options(ScaleFactor = 2)

if __name__ == "__main__":
    img_width, img_height = 384,384
    img_width_style, img_width_style = 256,256

    col1,col2 = st.columns(2)

    uploaded_file = col1.file_uploader("Choose the Image")
    if uploaded_file != None:
        content_image = load_image(uploaded_file,(img_width,img_height), col = col1)

    uploaded_file = col2.file_uploader("Choose the Image")
    if uploaded_file != None:
        style_image = load_image(uploaded_file,(img_width,img_height), col = col2)
        style_image = tf.nn.avg_pool(style_image, ksize = [3,3], strides = [1,1], padding = 'SAME')

        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]
        col3,col4,col5 = st.columns(3)
        col4.markdown('#The Output')
        show_image([stylized_image],title=['Stylized images'],col = col4)





