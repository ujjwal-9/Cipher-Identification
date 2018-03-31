# Cipher-Identification
Identification of Ciphers using Support Vector Machine and Neural Network.

- Identification using SVM can be found in [SVM.ipynb](https://github.com/Ujjwal-9/Cipher-Identification/blob/master/SVM.ipynb)
- Identification using NN can be found in [NN.ipynb](https://github.com/Ujjwal-9/Cipher-Identification/blob/master/NN.ipynb) 

We used tfidf of every character in cipher with respect to whole training dataset as our feature vector. In document categorization tasks, a document is considered as a bag of words and it is represented by a document vector whose elements correspond to the frequency of occurrence of different words in the document. The dimension of the document vector is the same as the size of the dictionary built by including all the distinct words that occur in a corpus of documents. Let N be the size of the dictionary. Let ti be the i th word or term in the dictionary, and tf(ti,d) be the frequency of occurrence of ti in a given document d. Then the N-dimensional document vector, φ(d), for a document d is given by: φ(d)=(t f(t1,d),t f(t2,d),...,t f(tN,d))TThis method of representing a text document by a document vector is called as the bag-of-words method. A term or word in a text document corresponds to a sequence of symbols or characters between two consecutive delimiters. we adopted the bag of-words method for representation of a ciphertext by a document vector. 


We feed these vector as our features to our support vector classifiers and neural networks.


- For support vector classifiers we use the following loss function:
`Loss_svc = max(0, 1 - y * w'x)`

- For neural network we used the following loss function:
`Binary crossentropy loss`


We did not use any validation set because only 100 training samples dataset were allowed to use. So due to shortage of training data we didn’t use validation set. We used accuracy as our metric to compare the performance as it gives us clear idea which model performed better when we have a balanced dataset.
