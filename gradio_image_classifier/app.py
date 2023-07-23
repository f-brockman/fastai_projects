
from fastai.vision.all import *

def is_cat(x): return x[0].isupper()

import gradio as gr

learn = load_learner('model.pkl')

categories = ('Dog','Cat')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return(dict(zip(categories,map(float,probs))))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()

intf = gr.Interface(fn = classify_image, inputs = image, outputs = label)
intf.launch(inline=False, share=True)