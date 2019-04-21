from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import pickle

max_length = 150
min_length = 10

X_train = pickle.load(open("./SST2/train_x","rb"))
train_y = pickle.load(open("./SST2/train_y","rb"))
X_test = pickle.load(open("./SST2/test_x","rb"))
test_y = pickle.load(open("./SST2/test_y","rb"))
X_val = pickle.load(open("./SST2/val_x","rb"))
val_y = pickle.load(open("./SST2/val_y","rb"))
length = pickle.load(open("./SST2/len","rb"))

model=Sequential()
model.add(Embedding(length+1, 64, input_length=max_length))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(50))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(X_train, train_y, validation_data=(X_val, val_y), batch_size=512, epochs=20)
scores = model.evaluate(X_test, test_y, verbose=0)
print('Test accuracy:', scores[1])
