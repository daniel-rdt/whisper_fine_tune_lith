import torch
from transformers import pipeline
import gradio as gr
import os
from pytube import YouTube
# from googletrans import Translator


# use model with improved learning rate which yielded better performance
pipe = pipeline(model="daniel-rdt/whisper-lt-finetune")

def transcribe_url_yt(yt_url):
    # load youtube video url
    video = YouTube(str(yt_url))
    # filter to include audio only
    audio = video.streams.filter(only_audio=True)[0].download("yt_audio")
    # get corresponding thumbnail and title
    thumbnail = video.thumbnail_url
    title = video.title
    # make text transcription using transformer pipeline
    text = pipe(audio)["text"]
    # remove stored audio file from disk
    os.remove(audio)
    # # text translation
    # translator = Translator()
    # translation = translator.translate(text, src='lt')

    return text, title, thumbnail #, translation.text

def transcribe_microphone(audio):
    text = pipe(audio)["text"]
    return text

with gr.Blocks() as url:
    with gr.Row():
        with gr.Column():
            yt_url = gr.Textbox(label="Paste YouTube link to e.g. lithuanian news video \n (recommended test link https://www.youtube.com/watch?v=9JsBg3pTp8I):")
            transcribe_btn = gr.Button(value="Transcribe")
        with gr.Column():
            news_text = gr.Textbox(label="Transcribed first paragraph of video:")
        # with gr.Column():
        #     translated_text = gr.Textbox(label="First paragraph translation into English:")
    with gr.Row():
        with gr.Column():
            yt_title = gr.Textbox(label="Title of the (News) Video")
        with gr.Column():
            yt_thumbnail = gr.Image()
    
    transcribe_btn.click(transcribe_url_yt, inputs=yt_url, outputs=[news_text, yt_title, yt_thumbnail])#, translated_text])

with gr.Blocks() as mike:
    with gr.Column():
        mike_input = gr.Audio(source="microphone", label="Speak into the microphone.", type="filepath")
        transcribe_btn = gr.Button(value="Transcribe")
    with gr.Column():
        text = gr.Textbox(label="Transcription:")
    # with gr.Column():
    #     translated_text = gr.Textbox(label="First paragraph translation into English:")

    transcribe_btn.click(transcribe_microphone, inputs=mike_input, outputs=[text])

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Whisper Model fine-tuned to transcribe Lithuanian
    Real time demo that can either transcribe from e.g. a Lithuanian News video YouTube URL to text or directly from microphone input.
    """
    )
    gr.TabbedInterface([url, mike], ["Transcribe Youtube-URL", "Transcribe Audio"])

demo.launch()