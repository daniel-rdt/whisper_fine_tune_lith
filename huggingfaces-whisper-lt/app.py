from transformers import pipeline
import gradio as gr

pipe = pipeline(model="Tomas1234/common_voice")  

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Small Lithuanian",
    description="Realtime demo for Lithuanian speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()