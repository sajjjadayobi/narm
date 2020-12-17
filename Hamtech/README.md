# Chat Bot

## Basic System
  - Question Classification (Intent Recognizer)
    - Name, Location, yes/no, etc
  - Qustion Generation
    - same Q with different words
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
  - an information retrieval System (search into passage)
    - with NER and search for keywords
    - or any type of information retrieval (rule base, Bert, etc)
  - Question Answering based on related information
    - end to end QA [library](https://github.com/deepset-ai/haystack)
    

## Related Works:
  - [AVA Paper](https://arxiv.org/pdf/2003.04987.pdf)
  - how does it [work?](https://demo.deeppavlov.ai/#/en/chat) 
  - everything about [QA](https://project-awesome.org/seriousran/awesome-qa)


## Task
  - Iran Paper Text Classification
    - I did not care about it :)
    - [colab](https://colab.research.google.com/drive/10qSVMohOoeMoJeQ8CoGe5bGu9-fJ2Gc2)
      - deal with small and bad dataset 

  - a little Masked Language Model (Bert)
    - on Persian Shahname (Just for Fun)
    - based on HuggingFace [colab](https://colab.research.google.com/drive/1NprZo5cNn-xaA3JRmyGbtlmdiv6fEPQo)
    
  - Intent Classification
    - the question is about what?
    - [My Perain Topics & Types Classifier](https://colab.research.google.com/drive/18uaGfsQuH1jo7OVyntRkkgrKj_mH55uq#scrollTo=AnxFwrUy2UKD)
    - [Original Dataset](https://github.com/AmirAhmadHabibi/TheSuperQuestionTypeTopicClassifier)
    
  - Incorporate Tabular Data With Transformers (multimodal learning)
    - as intput of Information Finding Model: [Bert Embedds, topics, types] !!! 
    - like this [work](https://www.kdnuggets.com/2020/11/tabular-data-huggingface-transformers.html)
  
  - Unsupervise Topic Modeling
    - [Topic2Vector](https://github.com/ddangelov/Top2Vec) 
    - [BerTopic](https://github.com/MaartenGr/BERTopic#toc)
    
  - Question Answering
    - idea based on Transformer Machine Translation
      - Vocab is all you need !!!
    - [PersianQA](https://colab.research.google.com/drive/1k2QNWqUnSb8C4kEweymX7TWRSF8stS4n#scrollTo=owDosTgZY_pS) without any QA Dataset in Persian !?
    - all models [pretrained](https://huggingface.co/models?filter=question-answering,multilingual)


## Resources
  - Stanford NLP lab
  - Transformers (attention is all you need) paper
  - HuggingFace Examples
  - and any good video about QA, ChatBot, LM, etc 
    - mostly on YouTube
    
    
## what things I did learn?
  - what is the structure of a normal QA, ChatBot
  - why Transformers are like Resnet in NLP 
    - Paper (Bert, Transformers, ULMfit, )
  - learn about Transformers library (Hugging Face)
  - and ... 
  
## is there any Question?
  
  
