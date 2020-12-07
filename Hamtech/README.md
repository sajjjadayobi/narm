# Chat Bot

## Basic System
  - Open Domain or Close Domain (task oriented bots)
  - Question Classification (Intent Recognizer)
    - Name, Location, yes/no, etc
    - I don't know (irrelevant/uncertain)
    - 
    - Qustion Generation
      - same Q with different words
      - Gnerative System or Synonyms
  - Information Finding
    - local database
    - online search
  - Anwser Generator
    - Exact Estraction (start token, end token)
    - Generate new answer
  - Examples
    - Joya
      - Web base information
      - wikipedia
      - 4 comman qustion
      

## Ideas 
  - open-domain: ”knowledgeable, and addictive”
  - closed-domain: ”accurate, reliable and efficient”
  - AVA Paper
    - the Intent Classification model and the Sentence Completion model
    - find incorrect(Out of Vocab) words od Sentence Completion like BERT fill mask model
      - use for generating new question
    - Train BERT based on Bayesian neural networks
    - dropout-based Bayesian approximation (learn different model at the same time)
    - Quality of Dataset is the key to success (remove noise)
    - Data Collection: phone conversation between expert agents & Customer
      - 22,630 questions are selected and classified to 381
      - 17,395 irrelevant questions
      - 5 label for each question
      - 2 types of irrelevant questions: out of scope, unsuitable to be processed by Chatbot
        - unsuitable to be processed by Chatbot: need to more dialogs
     - Methods for better Intent Classification (On Top Of Bert)


## Task
  - Intent Classification
    - [perain topics & types classifier](https://colab.research.google.com/drive/18uaGfsQuH1jo7OVyntRkkgrKj_mH55uq#scrollTo=AnxFwrUy2UKD)
  - Question Answering
  - if I'll have time 
    - Spell Checker
    - Question Generation
