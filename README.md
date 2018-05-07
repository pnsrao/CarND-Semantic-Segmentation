# Semantic Segmentation

In this project, we use the semantic segmentation approach to classify each pixel of an image with a binary label, either as a "driveable road area" or not as one. As described in the class lectures, this uses a Fully Convolutional Neural Network.

## Architecture
A pre-trained VGG16 network is modified into a fully convolutional network as described in the class and in the code walkthrough.
The last fully connected layer is converted into a 1x1 convolution, which is then upsampled to get the same resolution as the original image. Depths of the layer is set equal to the number of classes.

"Skip Connections" aspect is used to improve performance for the final higher resolution image. Earlier pooled layers of VGG16 (Layer 3 and Layer 4) are added to appropriately upsampled versions of the final layer. Before addition, these are converted with
1x1 convolutions and their depths are made consistent with the number of classes (2).

In all cases, upsampling is achieved through the transpose convolution instruction.

## Cost function, optimization and training
The cross entropy softmax cost function is used. An Adam optimizer is used to minimize the cost funtion. In each layer (1x1 and transpose convolutions), L2 regularization is used.

After experimenting with various values, training is done with a batch size of 5, 50 epochs, a learning rate of 0.0009 and a L2 regularization value of 1e-3

## Loss progression
For the above setting, the loss progression was as follows
<pre>
  EPOCH:  0  avg_loss =  6.99471358904
  EPOCH:  1  avg_loss =  0.624376026721
  EPOCH:  2  avg_loss =  0.564036843078
  EPOCH:  3  avg_loss =  0.482226594769
  EPOCH:  4  avg_loss =  0.3924064739
  
  EPOCH:  9  avg_loss =  0.201322492084
  EPOCH:  19  avg_loss =  0.115496740387
  EPOCH:  29  avg_loss =  0.0752757519227
  EPOCH:  39  avg_loss =  0.0504242753417
  EPOCH:  49  avg_loss =  0.0371962225643
 </pre>
  
## Results
The results are in the runs/1525663704.9633374 directory checked into github
Performance seems to be reasonable with most of the road area marked in green. There seems to be additional highlighting of non road areas as green in some images but these don't seem to exceed 20% of the overall road arhttps://en.support.wordpress.com/markdown-quick-reference/ea.

-----------------
## Instructions in the original README

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` ihttps://en.support.wordpress.com/markdown-quick-reference/s not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
