## CSCE-633---Machine-Learning-Project

### 1) LSTM

Consists of:

      1) Data Preprocessing
      2) Data Encoding
      3) LSTM
      4) Evaluation
      
LSTM Architecture

  Embeding layer
  LSTM
  LSTM
  DENSE
  
Accuracy:
      
      80.076% when using indexes as word values
      87.23% when using TF-IDF values as word values
  
----------------------------------------------------------
  
### 2) C-RNN

We train a character-RNN (using mLSTM units) over training data and use the char-RNN as the feature extractor for sentiment analysis.

The method gives us an accuracy of 91.6%.

############################

300.982 seconds to transform 6920 examples <br />
40.644 seconds to transform 872 examples <br />
78.165 seconds to transform 1821 examples <br />
91.76 test accuracy <br />
00.25 regularization coef <br />
00139 features used <br />

############################

----------------------------------------------------------
  
