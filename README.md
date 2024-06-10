# PaliGemma implemented from scratch in PyTorch
Implementation of PaliGemma and all its pieces from scratch in PyTorch.

The idea is to have a clean, understandable and customizable codebase for PaliGemma and its components without the overhead of using a big library like HuggingFace's transformers. As PaliGemma is composed of visual encoder transformer (ViT/SigLIP) and language model decoder (Gemma), this repository contains the implementation of both ViT and Gemma. 

Each big component (ViT, Gemma, PaliGemma) is first implemented separately in Jupiter notebooks for better understanding and then translated to python scripts. Each component can load pre-trained weights from HuggingFace's transformers library.

My personal goal for this project is to understand the inner workings of these models and to be able to customize them to my needs.

## Literature (for better understanding)
 - [ViT](https://arxiv.org/abs/2010.11929)
 - [SigLIP](https://arxiv.org/abs/2303.15343)
 - [Gemma](https://arxiv.org/abs/2403.08295)
 - [Pali-3](https://arxiv.org/abs/2310.09199)
 - [PaliGemma](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md)
 - Other related papers:
    - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    - [Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5)
    - [Root Mean Square Layer normalization](https://arxiv.org/abs/1910.07467)
    - [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415v5)

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

The ViT model is simpler than Gemma and could be implemented without defined submodules, as a single class, and still be understandable. Gemma, on the other hand, is more complex and is implemented as a class with submodules. PaliGemma is a combination of the two models and is implemented as a class with submodules. 

## TODOs
- [x] Implement ViT/SigLip
    - [x] notebook
    - [x] python script
- [x] Implement Gemma
    - [x] notebook
    - [x] python script
- [-] Implement PaliGemma
    - [-] notebook
    - [ ] python script
- [ ] Optimize code
- [ ] Finetune PaliGemma
    - [ ] notebook
    - [ ] python script

## Acknowledgements
This project is inspired by the amazing [llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) repo, go check it out.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
