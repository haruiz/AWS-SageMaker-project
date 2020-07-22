# Udacity Nanodegree Project - AWS SageMaker Project 

Deploying a Pytorch model in Amazon Sagemaker using aws Lambda functions, and aws Api gateway 

# Dataset

### Large Movie Review Dataset 


Large Movie Review Dataset
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. See the README file contained in the release for more details.

> Maas, Andrew L., et al. [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/). In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_. Association for Computational Linguistics, 2011.

# Model

Since this project's objective was to learn how to deploy PyTorch models on AWS using SageMaker, the model presented is a basic example. 

```python
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())
```


## Some Util functions

```python
from sagemaker.pytorch import PyTorch
import sagemaker

# create bucket
sagemaker_session = sagemaker.Session()

# upload data
bucket = sagemaker_session.default_bucket()
input_data = sagemaker_session.upload_data(
        path="./data/", 
        bucket=bucket, 
        key_prefix="my_model")

# get excution role
role = sagemaker.get_execution_role()
# create estimator
estimator = PyTorch(entry_point="train.py",
                    source_dir="train",
                    role=role,
                    framework_version='0.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.m4.xlarge',
                    hyperparameters={
                        'epochs': 10,
                        'hidden_dim': 200,
                    })
```

To get more information about the SageMaker APi check this link [Amazon SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/)

