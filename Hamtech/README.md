# <center>Sajjad Ayoubi</center>
  
- my English is not great but you can understand it 
  -  I wanted to improve my English that's why I write my report in English
- however, I'm just young, take it easy :)


## Basic Structure for Chatbot (in my research)
  - Question Classification (Intent Recognizer)
    - what is the type of question
      - Name, Location, yes/no, etc
  - Question Generation
      - generate more than one question for better search
    - you can try the same Q (in meaning) with different words 
  - Information Finding (finding relevant information based on Question)
    - you can have a local database (of information about the common question in your company) 
    - or use an online search (for example in Wikipedia)
  - Answer Generator
    - Exact Extraction (start token, end token)
      - find the answer directly from the related passage  
    - Generate a new answer (with language model)
      - not directly from the passage 


## things that I think we need (just after research):

  - We need something like SQuAD
      - SQuAD is a Question Answering dataset which
          - has Wikipedia articles with some question about that page
          - answers come exactly from articles   
  - Topic modeling 
    - what is the Question about?
    - for search in information dataset
  - an information retrieval System (search into passage)
    - with NER and search for keywords
    - or any type of information retrieval (rule base, Bert, etc)
  - or we can use open source Question Answering based on related information
    - like this: an end to end QA [library](https://github.com/deepset-ai/haystack)

## Related Works:

  - AVA Paper [arxiv](https://arxiv.org/pdf/2003.04987.pdf)
      - a close domain chatbot for a financial company
  - everything about Chatbots [QA](https://project-awesome.org/seriousran/awesome-qa)
      - best resource for search in papers, codes, etc on Chatbot 


## Task

  - Iran Paper Text Classification (I forgot what was the name of the website !!!)
    - I did not care about it 🤔
    	- deal with small and bad dataset 
    - [Colab](https://colab.research.google.com/drive/10qSVMohOoeMoJeQ8CoGe5bGu9-fJ2Gc2)
      
      
  - a little Masked Language Model (Bert)
    - on Persian Shahname (I just test it, it's not a good LM)
    - [Colab](https://colab.research.google.com/drive/1NprZo5cNn-xaA3JRmyGbtlmdiv6fEPQo)
    	- v1: based on HuggingFace library
        - v2: from scratch with keras
        
  - Intent Classification
    - the question is about what?
    - [My Perain Topics & Types Classifier](https://colab.research.google.com/drive/18uaGfsQuH1jo7OVyntRkkgrKj_mH55uq)
    - [Original Dataset](https://github.com/AmirAhmadHabibi/TheSuperQuestionTypeTopicClassifier)
    
    
    
  - Question Answering
    - idea based on Transformer Machine Translation
      - Vocab is all you need !!!
 	  - From Germany to French
    - [PersianQA](https://colab.research.google.com/drive/1k2QNWqUnSb8C4kEweymX7TWRSF8stS4n) without any QA Dataset in Persian !?



## Resources that I used
  - Stanford NLP lab
  - papers
     - Joya (a persian QA system)
     - Transformers (attention is all you need)
     - Bert
     - SQuAD v2.0
  - HuggingFace Examples
  - Keras, PyTorch, TF websites
  - and any good video about QA, ChatBot, LM, etc 
    - mostly on YouTube

    

## what have I learned?

  - what is the structure of a normal QA, ChatBot
  - why Transformers are like Resnet in NLP 
    - Paper (Bert, Transformers, ULMfit, ULMo, ...)
  - learn about Transformers library (Hugging Face)
  - a little bit about Docker 😎
  - and many other things about NLP (in practice)

## Question?
	- 🧐
  
# <center><b>Thanks for your attention</b></center>
## <center>don't forget, attention is all you need 🤗</center>
