import streamlit as st
from PIL import Image
from app_helper import *

st.set_page_config(layout='wide')

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1491895200222-0fc4a4c35e18?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1267&q=80");
background-size: cover;
}
</style>
'''



st.markdown(page_bg_img, unsafe_allow_html=True)


st.title("Neural Style Transfer")
st.text("Streamlit app based on A Neural Algorithm of Artistic Style (Leon A. Gatys et. al 2015) ")

st.sidebar.header("Options")
st.sidebar.text("")
st.sidebar.text("")
ep = st.sidebar.slider("Number of Epochs ", 0, 20) 

st.text("")
st.text("")
col_left, _, col_right, _ = st.beta_columns(4)



col_left.header("Content Image")
content_image_ph = col_left.empty()
content_file = col_left.file_uploader("choose an content image ....", type=["jpg", "png"])

if content_file is not None:
    image_c = Image.open(content_file)
    content_image_ph.image(image_c, height=200, width =400)
    content_targets = extractor(load_img(image_c))['content']
    



col_right.header("Style Image")
style_image_ph = col_right.empty()
style_file = col_right.file_uploader("choose an style image ....", type=["jpg", "png"])

if style_file is not None:
    image_s = Image.open(style_file)
    style_image_ph.image(image_s, height=200, width=400)
    style_targets = extractor(load_img(image_s))['style']
    


_, col2, _ = st.beta_columns(3)
if (content_file is not None) and (style_file is not None):
    image = tf.Variable(load_img(image_c))
    
    content_targets = extractor(load_img(image_c))['content']
    style_targets = extractor(load_img(image_s))['style']
    
    col2.text("")
    col2.text("")
    col2.header("Resulted Image")
    result_image_ph = col2.empty()

    epochs = 5
    steps_per_epoch = 10
    
    weights = (1e3, 1e-1)
    
    progress_bar = col2.progress(0)
    status_text = col2.empty()
    
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
        
            perctage = int((step)/(epochs*steps_per_epoch) *100)
            progress_bar.progress(perctage)
            status_text.text("%d percent completed..." % perctage)
            train_step(image, (style_targets, content_targets), weights)
    
    status_text.text("Completed!!!")
    result_image_ph.image(tensor_to_image(image), height=300, width=500)
    





