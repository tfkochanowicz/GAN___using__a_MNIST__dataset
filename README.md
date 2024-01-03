Overview: This is a project I made for a Machine Learning class. This project uses a General Adversarial Network(GAN) to create images that look like human hand writing. The GAN model is still pretty new,
as it was created in 2014 by Ian Goodfellow, along with some other people. This model is used in mamy of the modern "AI" Art generation programs. This program was was ran with python 3.7 and uses some specific libraries from keras(which the newer version would be tensorflow.keras), 
along with numpy.

How do I run the app?: I developed and trained this model with the Free version of Pycharm from Jetbrains. This application requires a lot of processing power from the CPU or GPU. Even with a new CPU(2023) this application
will still require a significant amount of time to train. If you have access to a modern high end GPU such as a Nvidia 4090 ti this application will run in a fraction of the amount of time. If you have an Nvidia
GPU with the CUDA toolkit configured that is.

What does this app do, and how does it work?: A simplified description of how this works would be that this project has prewritten images that are from an existing dataset, so there is a model that tries to create these images and this model is competing against a different model that determines if these images 
are "real" or not. After running through a predetermined amount of cycles the model will be able to generate handwritten numbers that look similar to the MNIST dataset images. The implications of this model are pretty impactful
as they can create images based on set of existing images. However these images are not purely original since they are based on the dataset.

Any Questions?: Email me at tfkochanowicz@gmail.com
