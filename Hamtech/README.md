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
  
  
## Presentation:
  - We need something like SQuAD 2.0
    - Passage - common Questions with Answers from that Organization 
  - Topic modeling 
    - what is the Question about?
  - a information retrieval Sysem (search into passage)
    - with NER and search for key words
    - or any type of information retrieval (rule base, Bert, etc)
  - Question Answering based on related information
    - end to end QA [library](https://github.com/deepset-ai/haystack)
    

## Related Works:
  - [AVA Paper](https://arxiv.org/pdf/2003.04987.pdf)
  - open source [awesome QA](https://haystack.deepset.ai/docs/intromd)
  - how does it [work?](https://demo.deeppavlov.ai/#/mu/textqa) 
  - everything about [QA](https://project-awesome.org/seriousran/awesome-qa)


## Task
  - Iran Paper Text Classification
    - I did not care about it :)
    - [colab](https://colab.research.google.com/drive/10qSVMohOoeMoJeQ8CoGe5bGu9-fJ2Gc2)


  - a little Masked Language Model (Bert)
    - on Persian Shahname (Just for Fun)
    - based on HuggingFace [colab](https://colab.research.google.com/drive/1NprZo5cNn-xaA3JRmyGbtlmdiv6fEPQo)

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
    - [PersianQA](https://colab.research.google.com/drive/1k2QNWqUnSb8C4kEweymX7TWRSF8stS4n#scrollTo=owDosTgZY_pS) without any QA Dataset in Persian !?
    - all models [pretrained](https://huggingface.co/models?filter=question-answering,multilingual)
