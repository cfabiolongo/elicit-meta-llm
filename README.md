# EXclusive-AutoRegressive (EXAR) fine-tuning/prompting

This is the repository of the Python (3.10+) implementation for the Exclusive-Autoregressive fine-tuning/prompting of LLama-chat, for metacognitive LLMs building focused on Question-Answering.

![Image 1](https://github.com/cfabiolongo/elicit-meta-llm/blob/master/images/meta-build.jpg)

# Installation

### Pandas 

```sh
> pip install pandas
> pip install openpyxl
```

### Pytorch

Follow the instructions reported [here](https://pytorch.org/) for the current system.

### Llama 2 

* Download Llama-2-7b-chat-hf (or 70b) from [huggingface](Llama-2-7b-chat-hf) and copy it in a local folder. 

### QLoRA

```sh
> pip install transformers==4.34.0
> pip install peft==0.4.0
> pip install sentencepiece==0.1.99
> pip install datasets==2.13.0
> pip install accelerate==0.23.0
> pip install bitsandbytes==0.41.1
> pip install trl==0.4.7
> pip install safetensors>=0.3.1
> pip install scipy
```

# Code usage

This repository contains source code splitted in the following steps:

* *stage-zero* fine-tuning: 
* *stage-zero* evaluation
* *stage-one* DATASET+ building
* *stage-one* DATASET+ annotation
* *stage-one* meta-validator fine-tuning
* *stage-one* meta-validator evaluation
* *stage-one* EXAR fine-tuning
* *stage-one* EXAR evaluation 
* merged *stage-one* meta-validator evaluation
* merged *stage-one* EXAR evaluation 




