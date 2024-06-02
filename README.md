# PaliGemma implemented from scratch in pytorch
Implementation of PaliGemma and all its pieces from scratch in pytorch.

The idea is to have a clean, understandable and customizable codebase for PaliGemma and its components without the overhead of using a big library like HuggingFace's transformers. As PaliGemma is composed of visual encoder transformer (ViT/SigLIP) and language model decoder (Gemma), this repository contains the implementation of both ViT and Gemma. 

Each big component (ViT, Gemma, PaliGemma) is first implemented separately in Jupiter notebooks for better understanding and then translated to python scripts. Each component can load pre-trained weights from HuggingFace's transformers library.

My personal goal for this project is to understand the inner workings of these models and to be able to customize them to my needs.

## Literature (for better understanding)
 - [ViT]()
 - [SigLIP]()
 - [Gemma]()
 - [Pali-3]()
 - [PaliGemma]()

## Installation
You should know the drill. Clone the repository, create a virtual environment and install the requirements.
Play around with CUDA issues and so on :) 
```bash
pip install -r requirements.txt
```

## Usage
The notebooks are the best place to start. They contain the implementation of the models and some examples of how to use them. The python scripts are the same as the notebooks but without the intermediate steps and explanations.

My advice is the following:

    1. Start with the ViT notebook
    2. Continue with the Gemma notebook
    3. Combine the two in the PaliGemma notebook
    4. Check out python scripts
    5. Play around with the models
    6. Contribute to the project

## TODOs
- [x] Implement ViT
    - [x] notebook
    - [ ] python script
- [x] Implement Gemma
    - [x] notebook
    - [ ] python script
- [ ] Implement PaliGemma
    - [ ] notebook
    - [ ] python script
- [ ] Optimize code
- [ ] Finetune PaliGemma
    - [ ] notebook
    - [ ] python script
    