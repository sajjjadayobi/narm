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
  - a information retrieval Sysem (seach into passage)
    - with NER and search for key words
    - or any type of information retrieval (rule base, Bert, etc)
    
  - deeppavlov Approach !!!
    - how can we create a new thing like this? (for better performance)
    - how can we just use this?
  
  - Create a Demo with(Persian)
    - train a multikingual cased on SQuAD (fine-tune)
    - train ParsBert on translate of SQuAD (fine-tune)
    
    

## Related Works:
  - [AVA Paper](https://arxiv.org/pdf/2003.04987.pdf)
  - open source [awesome QA](https://haystack.deepset.ai/docs/intromd)
  - how does it [work?](https://demo.deeppavlov.ai/#/mu/textqa) 
  - everything about [QA](https://project-awesome.org/seriousran/awesome-qa)
  - My [Plan]()
   
   


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
    - all multilingual [pretrained](https://huggingface.co/models?filter=question-answering,multilingual) and fine-tuned on [XSQuAD](https://github.com/deepmind/xquad)
      
  - if I'll have time 
    - Spell Checker
    - Question Generation
