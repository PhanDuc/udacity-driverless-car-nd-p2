**# udacity-driverless-car-nd-p2**

This is my work for the Self-Driving Car Engineer Nanodegree ND-013 course project, "2.Traffic Sign Classifier" . The project problem can be found here: [https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)

---
## Basic solution
Please see the "basic" folder. 

Let us look at the data (the second column is the mean image of each class):
![image](data/data_summary.jpg)


_solution using LeNet .... to be updated ...._
---
## Advance solution
Please see the "advance" folder. 

There is some write up at my blog: [https://medium.com/@hengcherkeng/updated-my-99-40-solution-to-udacity-nanodegree-project-p2-traffic-sign-classification](https://medium.com/@hengcherkeng/updated-my-99-40-solution-to-udacity-nanodegree-project-p2-traffic-sign-classification-5580ae5bd51f#.da1xutkw2)

I use modified densenet[1] and obtained **99.40%** on the test set. 
The network complexity is about **27.0 million MAC** (multiply–accumulate operation counts). 

Here is my network structure. Each "Dense block" consists of concatenation of convolutions (in conv-bn-relu). Note that unlike the paper, dropout is not applied in the block. Instead, I use droupout after the block.

![image](advance/docs/images/003.png)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  ![image](advance/docs/images/002.png)

Here is the MAC computation

![image](advance/docs/images/001.png)

Finally, the loss curves are shown below.

![image](advance/docs/images/100.png)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  ![image](advance/docs/images/000.png)

### [Reference]
[1] "Densely Connected Convolutional Networks" - Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten, Arxiv 2016
