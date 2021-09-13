import os
import webbrowser

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

model_path = os.path.join('model', 'trained_model.h5')


@st.cache
def model_load():
    model = tf.keras.models.load_model(model_path)
    return model


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def loadtest(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = (tf.cast(image, tf.float32) / 255.0 * 2) - 1
    image = tf.image.resize(image,
                            [256, 256],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.expand_dims(image, 0)
    return image


def loadframe(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = (tf.cast(image, tf.float32) / 255.0 * 2) - 1
    image = tf.image.resize(image,
                            [256, 256],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.expand_dims(image, 0)
    return image


def cs_sidebar():
    st.sidebar.header('Navigation')
    current_page = st.sidebar.radio("", ["Image Colorization", "Documentation", "Team"])
    return current_page


def cs_image_colorization():
    st.markdown("<h1 style='text-align: center; color: white;'>Image Colorization</h1>",
                unsafe_allow_html=True)

    # comic_model = model_load()

    outputsize = 512
    gamma = 1.0

    Image = st.file_uploader('Upload grayscale image here', type=['jpg', 'jpeg', 'png'])
    my_expander = st.beta_expander(label='🙋 Upload help')
    with my_expander:
        st.markdown('Upload the grayscale image')
        st.markdown('Filetype to upload : **JPG, JPEG, PNG**')
    if Image is not None:
        col1, col2 = st.beta_columns(2)
        Image = Image.read()
        Image = tf.image.decode_image(Image, channels=3).numpy()
        Image = adjust_gamma(Image, gamma=gamma)
        with col1:
            col1.subheader("Uploaded Image")
            st.image(Image)
        input_image = loadtest(Image)
        # prediction = comic_model(input_image, training=False)
        # prediction = tf.squeeze(prediction, 0)
        # prediction = prediction * 0.5 + 0.5
        # prediction = tf.image.resize(prediction,
        #                              [outputsize, outputsize],
        #                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # prediction = prediction.numpy()
        with col2:
            col2.subheader("Colorized Image")
            # st.image(prediction)

    return None


def cs_documentation():
    st.markdown("<h1 style='text-align: center; color: white;'>Documentation</h1>",
                unsafe_allow_html=True)
    return None


def cs_team():
    st.markdown("<h1 style='text-align: center; color: white;'>Team</h1>",
                unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.beta_columns(5)

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
        st.write(name[0])
        click1 = st.button('Github', key="1")
        if click1:
            webbrowser.open_new_tab(git_url[0])
        click6 = st.button('Linkedin', key="6")
        if click6:
            webbrowser.open_new_tab(linkedlin_url[0])
    with col2:
        st.image(os.path.join('Images', 'hemanth.png'), use_column_width=True)
        st.write("Hemanth Reddy")
        click2 = st.button('Github', key="2")
        if click2:
            webbrowser.open_new_tab(git_url[1])
        click7 = st.button('Linkedin', key="7")
        if click7:
            webbrowser.open_new_tab(linkedlin_url[1])
    with col3:
        st.image(os.path.join('Images', 'brinda.png'), use_column_width=True)
        st.write("Brinda Potluri")
        click3 = st.button('Github', key="3")
        if click3:
            webbrowser.open_new_tab(git_url[2])
        click8 = st.button('Linkedin', key="8")
        if click8:
            webbrowser.open_new_tab(linkedlin_url[2])
    with col4:
        st.image(os.path.join('Images', 'pranathi.png'), use_column_width=True)
        st.write("Pranathi Karedla")
        click4 = st.button('Github', key="4")
        if click4:
            webbrowser.open_new_tab(git_url[3])
        click9 = st.button('Linkedin', key="9")
        if click9:
            webbrowser.open_new_tab(linkedlin_url[3])
    with col5:
        st.image(os.path.join('Images', 'praveen.png'), use_column_width=True)
        st.write("Praveen Kumar")
        click5 = st.button('Github', key="5")
        if click5:
            webbrowser.open_new_tab(git_url[4])
        click10 = st.button('Linkedin', key="10")
        if click10:
            webbrowser.open_new_tab(linkedlin_url[4])

    return None


def main():
    st.set_page_config(
        page_title='Image Colorization',
        layout="wide",
        initial_sidebar_state="expanded",
    )

    current_page = cs_sidebar()
    if (current_page == "Image Colorization"):
        cs_image_colorization()
    elif (current_page == "Documentation"):
        cs_documentation()
    else:
        cs_team()
    return None


if __name__ == '__main__':
    main()
