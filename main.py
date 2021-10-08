import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from skimage import color

model_path = os.path.join('model', 'trained_model.h5')


@st.cache(allow_output_mutation=True)
def model_load():
    model = tf.keras.models.load_model(model_path)
    return model


def rgb_to_lab(img, l=False, ab=False):
    """
    Takes in RGB channels in range 0-255 and outputs L or AB channels in range -1 to 1
    """
    img = img / 255
    l_chan = color.rgb2lab(img)[:, :, 0]
    l_chan = l_chan / 50 - 1
    l_chan = l_chan[..., np.newaxis]

    ab_chan = color.rgb2lab(img)[:, :, 1:]
    ab_chan = (ab_chan + 128) / 255 * 2 - 1
    if l:
        return l_chan
    else:
        return ab_chan


def lab_to_rgb(img):
    """
    Takes in LAB channels in range -1 to 1 and out puts RGB chanels in range 0-255
    """
    new_img = np.zeros((256, 256, 3))
    for i in range(len(img)):
        for j in range(len(img[i])):
            pix = img[i, j]
            new_img[i, j] = [(pix[0] + 1) * 50, (pix[1] + 1) / 2 * 255 - 128, (pix[2] + 1) / 2 * 255 - 128]
    new_img = color.lab2rgb(new_img) * 255
    new_img = new_img.astype('uint8')
    return new_img


def cs_sidebar():
    st.sidebar.header('Navigation')
    current_page = st.sidebar.radio("", ["Image Colorization", "Team"])
    return current_page


def cs_image_colorization():
    st.markdown("<h1 style='text-align: center; color: white;'>Image Colorization</h1>",
                unsafe_allow_html=True)

    Image = st.file_uploader('Upload grayscale image here', type=['jpg', 'jpeg', 'png'])
    my_expander = st.expander(label='🙋 Upload help')
    with my_expander:
        st.markdown('Upload the grayscale image')
        st.markdown('Filetype to upload : **JPG, JPEG, PNG**')
    if Image is not None:
        col1, col2 = st.columns(2)
        Image = Image.read()
        Image = tf.image.decode_image(Image, channels=3).numpy()
        height = Image.shape[0]
        width = Image.shape[1]
        with col1:
            col1.subheader("Uploaded Image")
            st.image(Image)
            model = model_load()
            Image = tf.image.resize(Image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            l_channel = rgb_to_lab(Image, l=True)
            fake_ab = model.predict(l_channel.reshape(1, 256, 256, 1))
            fake = np.dstack((l_channel, fake_ab.reshape(256, 256, 2)))
            fake = lab_to_rgb(fake)
            fake = cv2.resize(fake, (width, height), interpolation=cv2.INTER_AREA)
        with col2:
            col2.subheader("Colorized Image")
            st.image(fake)

    return None


def cs_team():
    st.markdown("<h1 style='text-align: center; color: white;'>Team</h1>",
                unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)

    name = ["Aditya Pujari",
            "Hemanth Reddy",
            "Brinda Potluri",
            "Pranathi Karedla",
            "Praveen Kumar"]

    git_url = ["https://github.com/adityapujari98",
               "https://github.com/HemanthReddy09",
               "https://github.com/brindapotluri",
               "https://github.com/pranathi1997",
               "https://github.com/praveensomara"]

    linkedlin_url = ["https://www.linkedin.com/in/adityapujari2/",
                     "https://www.linkedin.com/in/hemanth-reddy-6b72a0158/",
                     "https://www.linkedin.com/in/brinda-potluri-96009020a/",
                     "https://www.linkedin.com/in/pranathikaredla/",
                     "https://www.linkedin.com/in/praveenkumarsomara/"]

    with col1:
        st.image(os.path.join('Images', 'aditya.png'), use_column_width=True)
        st.markdown('### ' + name[0])
        github_link = '[Github](' + git_url[0] + ')'
        st.markdown(github_link, unsafe_allow_html=True)
        linkedin_link = '[Linkedin](' + linkedlin_url[0] + ')'
        st.markdown(linkedin_link, unsafe_allow_html=True)
    with col2:
        st.image(os.path.join('Images', 'hemanth.png'), use_column_width=True)
        st.markdown('### ' + name[1])
        github_link = '[Github](' + git_url[1] + ')'
        st.markdown(github_link, unsafe_allow_html=True)
        linkedin_link = '[Linkedin](' + linkedlin_url[1] + ')'
        st.markdown(linkedin_link, unsafe_allow_html=True)
    with col3:
        st.image(os.path.join('Images', 'brinda.png'), use_column_width=True)
        st.markdown('### ' + name[2])
        github_link = '[Github](' + git_url[2] + ')'
        st.markdown(github_link, unsafe_allow_html=True)
        linkedin_link = '[Linkedin](' + linkedlin_url[2] + ')'
        st.markdown(linkedin_link, unsafe_allow_html=True)
    with col4:
        st.image(os.path.join('Images', 'pranathi.png'), use_column_width=True)
        st.markdown('### ' + name[3])
        github_link = '[Github](' + git_url[3] + ')'
        st.markdown(github_link, unsafe_allow_html=True)
        linkedin_link = '[Linkedin](' + linkedlin_url[3] + ')'
        st.markdown(linkedin_link, unsafe_allow_html=True)
    with col5:
        st.image(os.path.join('Images', 'praveen.png'), use_column_width=True)
        st.markdown('### ' + name[4])
        github_link = '[Github](' + git_url[4] + ')'
        st.markdown(github_link, unsafe_allow_html=True)
        linkedin_link = '[Linkedin](' + linkedlin_url[4] + ')'
        st.markdown(linkedin_link, unsafe_allow_html=True)

    return None


def main():
    st.set_page_config(
        page_title='Image Colorization',
        layout="wide",
    )

    current_page = cs_sidebar()
    if (current_page == "Image Colorization"):
        cs_image_colorization()
    else:
        cs_team()
    return None


if __name__ == '__main__':
    main()
