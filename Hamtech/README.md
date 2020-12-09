# Chat Bot

## Basic System
  - Open Domain or Close Domain (task oriented bots)
  - Question Classification (Intent Recognizer)
    - Name, Location, yes/no, etc
    - I don't know (irrelevant/uncertain)
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
      - Web base information (wikipedia)
      - 4 comman qustion
      

## Ideas 
  - open-domain: ”knowledgeable, and addictive”
  - closed-domain: ”accurate, reliable and efficient”
  - [AVA Paper](https://arxiv.org/pdf/2003.04987.pdf)
    - the Intent Classification model and the Sentence Completion model
    - find incorrect(Out of Vocab) words od Sentence Completion like BERT fill mask model
      - use for generating new question
    - Data Collection: phone conversation between expert agents & Customer
      - 22,630 questions are selected and classified to 381
      - 17,395 irrelevant questions, 5 label for each question


## Task
  - Intent Classification
    - [My Perain Topics & Types Classifier](https://colab.research.google.com/drive/18uaGfsQuH1jo7OVyntRkkgrKj_mH55uq#scrollTo=AnxFwrUy2UKD)
    - [Original Dataset](https://github.com/AmirAhmadHabibi/TheSuperQuestionTypeTopicClassifier)
    
  - Incorporate Tabular Data With Transformers (multimodal learning)
    - as intput of Information Finding Model: [Bert Embedds, topics, types] !!! 
    - like this [work](https://www.kdnuggets.com/2020/11/tabular-data-huggingface-transformers.html)
  
  - Unsupervise Topic Modeling
    - [Topic2Vector](https://github.com/ddangelov/Top2Vec) 
    - [BerTopic](https://github.com/MaartenGr/BERTopic#toc)
    
  - information retrieval
  - Question Answering

  - if I'll have time 
    - Spell Checker
    - Question Generation
