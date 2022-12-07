
import torch
from transformers import pipeline
import gradio as gr
import os
from pytube import YouTube



# use model with improved learning rate which yielded better performance
pipe = pipeline(model="daniel-rdt/whisper-lt-finetune")

def transcribe_url_yt(yt_url):
    # load youtube video url
    video = YouTube(str(yt_url))
    audio = video.streams.filter(only_audio=True)[0].download("yt_audio")
    thumbnail = video.thumbnail_url
    title = video.title
    text = pipe(audio)["text"]
    os.remove(audio)
    return text, title, thumbnail

title = "Lithuanian News Whisperer"

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            yt_url = gr.Textbox(label="Paste YouTube link to lithuanian news video:")
            translate_btn = gr.Button(value="Transcribe")
        with gr.Column():
            news_text = gr.Textbox(label="First paragraph of transcibed News:")
    with gr.Row():
        with gr.Column():
            yt_title = gr.Textbox(label="Title of the News Video")
        with gr.Column():
            yt_thumbnail = gr.Image()

    translate_btn.click(transcribe_url_yt, inputs=yt_url, outputs=[news_text, yt_title, yt_thumbnail])

demo.launch()