
Recognizes handwritten Tamil characters with 90% accuracy. Credits to HP Labs India for the training and test datasets. This system uses a convolutional neural network (CNN), which is widely used across optical character recognition tasks.

Introduction
Tamil is language originally from the South Indian state of Tamil Nadu. It is predominantly used in South India, Sri Lanka, and Singapore. It is one of the oldest languages in the world and is spoken by over 80 million people worldwide. Tamil uses a non-Latin script; the alphabet consists of 156 characters, including 12 vowels and 23 consonants. Due to the large number of classes and the extreme similarity between certain characters, accurate Tamil character recognition is more challenging than standard Latin character recognition. As with any language, handwritten character recognition is useful in a wide range of applications, including the digitization of legal documents, mail sorting in post offices, and bank check reading.

Dataset
The dataset of offline handwritten Tamil characters is taken from HP Labs India. It contains approximately 500 examples of each of the 156 characters, written by native writers in Tamil Nadu, India. For the IWFHR 2006 Tamil Character Recognition Competition, the entire datset was split into separate training (50,683 examples) and test sets (26,926 examples), which were used here. The provided training set was subsequently split into a new training set and a validation set in a 80% to 20% ratio.

The bi-level images are initially provided as TIFF files of various sizes. After being converted to the PNG format, the images were inverted such that the foreground and background were white and black, respectively, and a constant thickening factor was applied. Then, the images were resized such that the longer side length was 48 pixels, using the Lanczos algorithm. The Lanczos algorithm applies anti-aliasing, causing the image to shift from bi-level to grayscale. Finally, the centers of mass of the resulting images were centered on a new 64 x 64 canvas. These images are normalized by transforming each grayscale pixel value from the [0, 1] range to the [-1, 1] range.

Architecture
The input is passed into the model as a 64 x 64 image. The model is structures as follows:
[1x64x64] INPUT
[16x64x64] CONV: 16 3x3 filters with stride 1, pad 1
[16x64x64] CONV: 16 3x3 filters with stride 1, pad 1
[16x32x32] MAX POOL: 2x2 filters with stride 2
[32x32x32] CONV: 32 3x3 filters with stride 1, pad 1
[32x32x32] CONV: 32 3x3 filters with stride 1, pad 1
[32x16x16] MAX POOL: 2x2 filters with stride 2
[64x16x16] CONV: 64 3x3 filters with stride 1, pad 1
[64x16x16] CONV: 64 3x3 filters with stride 1, pad 1
[64x8x8] MAX POOL: 2x2 filters with stride 2
[1024] FC: 1024 neurons
[512] FC: 512 neurons
[156] FC: 156 neurons (class neurons)

Every convolutional and fully connected layer is directly followed by batch normalization and a ReLU activation.

Experiments
Training
Training was done on a GPU via Google Colab. There were several hyperparameters to tune, including but not limited to learning rate, weight decay (L2 regularization penalty), and initialization. Throughout the process, I referred to the online Notes for CS231n at Stanford by Andrej Karpathy. I tested applying dropout on all layers as well as on only fully connected layers, but both configurations resulted in lower validation accuracy. Thus, an L2 penalty of 0.003 was chosen. All layers were initialized using Kaiming initialization and the optimizer of choice was Adam, with a learning rate of 0.001.

Testing
Testing was also conducted on a Google Colab GPU. The final model achieved 90.7% accuracy on the test set, which was satisfactory for me. As previously mentioned, since there are 156 classes, several of which are very similar to one another, attaining high accuracy is an especially difficult task. Test accuracy was consistently lower than validation accuracy, which suggests that the test set for the competition was deliberately made to be more difficult than the training set.
