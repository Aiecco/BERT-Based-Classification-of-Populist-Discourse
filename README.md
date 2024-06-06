# How to run the model
In order to run the model, you have to retrain the language classifier (RoBERTa) in your machine (or through a cloud GPU service like Colab) with the code found in the zip. The dataset is called "DB.xlsx", and since the code was run in Colab, if you want to run it locally, you have to adjust the path.

To predict the populist-ness of an input speech, run the following code after RoBERTa's training loop.


    def predict_speech(speech, model, tokenizer):
        inputs = tokenizer.encode_plus(
            speech,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
    
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
    
    
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(0)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_label = probabilities.argmax(dim=-1).item()
            confidence = probabilities.max().item()
    
    
        label_map = {0: "Not Populist", 1: "Populist"}
    
        print(f"Speech: {speech}")
    
        if confidence >= 0.88:
            print(f"This speech is highly populist (Confidence: {confidence:.2f}).")
        if confidence >= 0.70 and confidence <= 0.87:
            print(f"This speech is very likely populist (Confidence: {confidence:.2f}).")
        if confidence >= 0.55 and confidence <= 0.69:
            print(f"This speech is probably populist (Confidence: {confidence:.2f}).")
        if confidence >= 0.25 and confidence <= 0.54:
            print(f"This speech is possibly populist (Confidence: {confidence:.2f}).")
    
        if confidence <= -0.88:
            print(f"This speech is not populist at all (Confidence: {confidence:.2f}).")
        if confidence <= -0.70 and confidence >= -0.87:
            print(f"This speech is very likely not populist (Confidence: {confidence:.2f}).")
        if confidence <= -0.55 and confidence >= -0.69:
            print(f"This speech is probably not populist (Confidence: {confidence:.2f}).")
        if confidence <= -0.25 and confidence >= -0.54:
            print(f"This speech is possibly not populist (Confidence: {confidence:.2f}).")
    
        if confidence >= -0.24 and confidence <= 0.24:
            print(f"This speech is hard to label, but could contain some populist elements (Confidence: {confidence:.2f}).")

    user_speech = input("Enter a speech to classify: ")
    predict_speech(user_speech, model, tokenizer)

Example:

INPUT: 

"We the American people have been stripped of our values. Let's take them back and kill Joe Biden!"

OUTPUT:

This speech is highly populist (Confidence: 0.98).

# Binary Classification of Populist Speech
## Overview

This repository contains the code and data for the paper "Binary Classification of Populist Speech". The study addresses the classification of speeches as populist or non-populist using fine-tuned pre-trained language models. The models evaluated include BERT-tiny, BERT-large, GPT-2, and RoBERTa-large.
### Authors

    Alessandro Pala - alessandro.pala@studenti.unipd.it
    Lorenzo Cino - lorenzo.cino@studenti.unipd.it
    Greta Grelli - greta.damoregrelli@studenti.unipd.it
    Alberto Calabrese - alberto.calabrese2@studenti.unipd.it
    Giacomo Filippin - giacomo.filippin@studenti.unipd.it

## Abstract

This project focuses on classifying speeches as populist or non-populist using fine-tuned language models. We utilized a dataset of 500 labelled speeches, equally split between populist and non-populist, and evaluated four models: BERT-tiny, BERT-large, GPT-2, and RoBERTa-large. The RoBERTa-large model achieved the best performance with an accuracy of 88%.
### Table of Contents

    Introduction
    Dataset
    Methodology
        Preprocessing
        Model Fine-Tuning
    Experiments and Results
        BERT-tiny
        BERT-large
        GPT-2
        RoBERTa-large
    Conclusion
    Usage
    References

## Introduction

Populism is a significant issue in political speech. This project aims to fine-tune pre-trained language models to classify text as populist or non-populist. Populist speeches often mention "the people" as a unity against the "corrupt elite" and may employ fearmongering and anti-establishment rhetoric.
Dataset

The dataset consists of 500 speeches, with 250 populist and 250 non-populist speeches. Each speech is manually labelled. The data is in .xlsx format, with speeches in one column and labels (0 for non-populist, 1 for populist) in the adjacent column.
## Methodology - Preprocessing

    Data Splitting: The dataset is split into training and testing subsets (80-20 split).
    Tokenization: Each model's tokenizer converts the speeches into token IDs, adds special tokens, and pads/truncates sequences to a fixed length.
    Data Loading: Custom data loader classes create iterators for batching and shuffling the datasets.

## Model Fine-Tuning

Four pre-trained models were fine-tuned:

    BERT-tiny: Small encoder-only model by Google.
    BERT-large: Large encoder-only model by Google.
    GPT-2: Large decoder-only model by OpenAI.
    RoBERTa-large: Large encoder-only model by Facebook AI.

Training involved using the Adam optimizer with exponential weight decay and a dynamic learning rate scheduler. Gradient clipping was applied to prevent exploding gradients.

## Experiments and Results

BERT-tiny

    Accuracy: ~0.59
    Observations: Gradual decline in loss, but accuracy plateaued slightly above random.

BERT-large

    Accuracy: ~0.71
    Observations: Better than BERT-tiny, but still plateaued at an unsatisfactory rate.

GPT-2

    Accuracy: ~0.61
    Observations: Limited improvement in loss, noisy accuracy trends.

RoBERTa-large

    Accuracy: ~0.88
    Observations: Best performance among all models, demonstrating its suitability for the task.

## Conclusion

RoBERTa-large outperformed other models, achieving an accuracy of 88%. This suggests that large, encoder-only pre-trained models optimized for robust performance are highly suitable for binary classification tasks involving nuanced textual analysis.

References

    Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
    Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.
    Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
    Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models.

For more details, please refer to the full paper included in this repository.
