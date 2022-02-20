## DSFD: Dual Shot Face Detector ##
[A PyTorch Implementation of Dual Shot Face Detector](https://arxiv.org/abs/1810.10220?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)

### Description
I use basenet [vgg](https://pan.baidu.com/s/1Q-YqoxJyqvln6KTcIck1tQ) to train DSFD,the model can be downloaded in [DSFD](https://pan.baidu.com/s/17cpDHEwYVxWmOIPqUy5zCQ).the AP in WIDER FACE as following:  

| Easy MAP | Medium MAP	|  hard MAP |
| ---------|------------| --------- |
|	0.946  |    0.937   |  0.880    | 
 
the AP in AFW,PASCAL,FDDB as following:

| 	AFW     |   PASCAL	|   FDDB   |
| --------- |-----------| ---------|
|	99.89   |    99.11  |  0.983   |
 
I'm using resnet50/resnet101 to train DSFD,the result will be published later on 
### Requirement
* pytorch 0.3 
* opencv 
* numpy 

### Prepare data 
1. download WIDER face dataset
2. modify data/config.py 
3. ``` python prepare_wider_data.py```


### Train 
``` 
python train.py --batch_size 4 
		--model vgg\resnet50\resnet101 
		--lr 5e-4
``` 

### Evalution
test on WIDER FACE 
```
python test.py
```
### Demo 
you can test yourself image
```
python test_demo.py
```

### References
* [Dual Shot Face Detector](https://arxiv.org/abs/1810.10220?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)
* [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)