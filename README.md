# Lithuanian Whisper Fine Tune
Lab2: Lithuanian Text Transcription using Transformers

The second lab of the course ID2223 was intended to fine-tune the pre-trained transformer model whisper and build a serverless UI for using that model. The model is based on the blog ''Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers'' by Sanchit Gandhi.
The program architecture was built to be used serverless and in different pipelines in order to allow to run feature engineering on CPU performance and training on GPUs. The three-pipeline archtitecture consists of:

1. whisper_lt_feature_pipeline.ipynb
2. whisper_lt_training_pipeline.ipynb
3. huggingface-whisper-lt-finetune/app.py

The feature and training pipeline can either be run locally or using cloud computing services such as google-colab. Especially for the training pipeline, the use of a GPU cloud computing service is recommended.

## Feature Pipeline
In the `Feature Pipeline`, first the training and testing dataset `common voice` is imported. Additionally, the loaded dataset is transformed into the correct format for fine-tuning the whisper model. A feature extractor which pre-processes the raw audio-inputs is needed. For this purpose a so-called `WhisperProcessor` from HuggingFace transformers is utilized that can also be used as the tokenizer which post-processes the model outputs to text format later. The Lithuanian whisper-small checkpoint is used for this.

The prepared dataset is then uploaded to `Hopsworks` to be downloaded later in the training pipeline.

## Training Pipeline

In the `Training Pipeline` as a first step, it needs to be made sure that an appropriate GPU is being used or assigned respectively. Then the pre-processed common-voice dataset is loaded from hopsworks.
Then, to perform the model training, a datacollector needs to be defined. Again the WhisperProcessor is utilized here. As evaluation metric `Word Error Rate (WER)` is used. The from the `whisper-small` checkpoint the model is trained using the `Seq2SeqTrainer` from HuggingFace transformers.

There are two different approaches that can be used to improve model performance when training the model:

### Model-centric approach
The first approach to increase model performance would be to fine-tune the models hyperparameters. As for every ML model, the correct choice of hyperparameters is essential to achieving high model performance. One simple solution is to change the learning rate for training the model while keeping the number of steps fixed. By increasing the learning rate slightly it is to be expected that the model could improve further in the later checkpoints of the trainig process.

It was possible to varify this apporach. As expected, with an increased learning-reate from 1e-5 to 5e-5, the model shows worse WER performance in the earlier checkpoints but catches up during the later steps of the training. Compared to the base set-up the best model WER improved from 32.4971 to 28.1159 WER. 


The improved model (which will be further used) and the used hyperparamters can be found here: https://huggingface.co/daniel-rdt/whisper-lt-finetune

The base model and the used hyperparameters can be found here: https://huggingface.co/Tomas1234/common_voice

### Data-centric approach
The second approach would be to utilize a different datasource. Clearly utilizing a greater amount of training data is very probable to yield better model performance. For excample, instead of fine-tuning the model on the `whisper-small` checkpoint of the whisper model, one could utilize the greater checkpoint of `whisper-medium`. Since this comes with considerably larger computational times however, within the scope of this lab assignment, this approach could not be varified. 

## UI Inference
The UI Inference demo can be found here: https://huggingface.co/spaces/daniel-rdt/whisper-lt-finetune

The aim of the UI is to demonstrate the value-added capabilities of the fine-tuned model. The user can paste any youtube url and prompt the model to transcribe the first few sentences of spoken lithuanian text in the video. Useful applications include the transcription of news feed as well as political speeches or even podcasts.
