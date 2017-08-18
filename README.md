# JavaScriptExtract
Pytorch implementation of a selective JavaScript autocompletion model. Tested on Python 3.5
- Requirements
  - PyTorch
  - numpy
  - astor
  - tensorflow, scipy (only needed for tensorboard Logger)

- usage: python main.py --mode=train
  - arguments:
  
    - mode = train / test
    - load = set to None, can be replaced with the location of a previously saved weight
    - copy = if True, copies all files into a temporary folder
    - encoder: method to encode lines for similarity measurement (either position or lstm) 
    - similarity: which similarity measurement to use (either cosine or mlp)
    - everything else is described at main.py
    
