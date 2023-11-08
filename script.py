import tensorflow as tf
import gradio as gr
import numpy as np
import os
import pandas as pd
from tensorflow.keras.layers import TextVectorization





df = pd.read_csv(os.path.join('train.csv','C:\Users\User\Desktop\P.F.A 4eme annee\train.csv'))
x= df['comment_text']
y= df[df.columns[2:]].values
MAX_WORDS = 500000
vectorizer = TextVectorization(max_tokens=MAX_WORDS,output_sequence_length=1800,output_mode='int')
vectorizer.adapt(x.values)

model = tf.keras.models.load_model('toxicity.h5')
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    text=''
    for idx, col in enumerate(df.columns[2:]):
        text +='{}:{}\n'.format(col, results[0][idx]>0.5)
    return text    
interface = gr.Interface(fn=score_comment, inputs=gr.inputs.Textbox(lines=2,placeholder='Comment to score'),outputs='text' )
interface.launch(share=True)