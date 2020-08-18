
![perceptron](https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/ArtificialNeuronModel_english.png/800px-ArtificialNeuronModel_english.png)

a simple multilayer NeuralNetwork for recognizing MNIST hand written digits, written using `numpy` library. this repo is to help beginners to understanding neuralnetwork stuff easier. [Here](https://github.com/kdexd/digit-classifier) you can find the optimised version.

### network structure

| Layers | Nodes | Weights | Biases |
| :---: | :---: | :---: | :--: |
| l0: input-layer | (1, 784) | -- | -- |
| l1: hidden-layer | (1, 300) | (300, 784) | (1, 300) |
| l2: hidden-layer | (1, 90) | (90, 300) | (1, 90) |
| l3: output-layer | (1, 10) | (10, 90) | (1, 10) |

*Note: these are example shapes, except no.of layers 

#### data set

I used dataset from kaggle website,
 1. Go [here](https://www.kaggle.com/c/digit-recognizer/data) and download.
 2. Extract the files, and rename the folder to `data`
 3. move `data` folder to this repo.

#### key points:

* Used SGD(Stochastic Gradient Descent) method for training.
* Available activation functions sigmoid, relu and softmax.
* trained session of network not implemented.
* able to see predicted digit.
* able to shuffle the data while training.
* kaggle submission file generation.

#### Execution

* Go to `src` folder using `cd src`
* run `python main.py`

### why this repo?

There are multiple repos out there for this problem, they are pretty much optimised and complex for beginners. So I made this repo for simple for beginners to understand basic operations like feedforward, backpropagations and simple tuning.

#### Lang\libraries\Tools
* `Python 3.8.3`
* `numpy`
* `pandas`
* `matplotlib`

### Acknowledgments
* [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* [yannlecun](http://yann.lecun.com/exdb/mnist/)
* [kaggle](https://www.kaggle.com/c/digit-recognizer)
* [Matt Mazur](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
* [iamtrask](https://iamtrask.github.io/2015/07/12/basic-python-network/)
* [kdexd](https://github.com/kdexd/digit-classifier)
* perceptron-image used in `readme.md` file from [Chrislb](https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/ArtificialNeuronModel_english.png/800px-ArtificialNeuronModel_english.png)



