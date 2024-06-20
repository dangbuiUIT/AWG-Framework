from sklearn.model_selection import train_test_split
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from keras.optimizers import SGD, Adam
import tensorflow as tf
from keras import layers, models
from keras.utils import plot_model

import data_process

df = pd.read_csv("/content/drive/MyDrive/Journal/dataset/shark_visual.csv")
benign = 9759
phish = len(df) - benign
l1 = []
l2 = []
l3 = []
#array_from_str = np.array(eval(first_element[0]))
for i in range(len(df)):
  try:
    element = df.iloc[i]
    array_from_str = np.array(eval(element[0]))
    l1.append(array_from_str.reshape((10,10,1)))

    array_from_str = np.array(eval(element[1]))
    l2.append(array_from_str)

    array_from_str = np.array(eval(element[2]))
    l3.append(array_from_str)
  except:
    print(i)
    continue

# generate labels
labels = np.concatenate([np.zeros(benign), np.ones(phish)])

#split dataset
imgl_train, imgl_test, doml_train, doml_test, strucl_train, strucl_test, labels_train, labels_test = train_test_split(
    l1, l3, l2, labels, test_size=0.3, random_state=25)

#define model
# Define the image input
image_input = tf.keras.Input(shape=(10, 10, 1), name='image_input')
conv1 = layers.Conv2D(64, (3, 3), activation=None)(image_input)
batch_norm1 = layers.BatchNormalization()(conv1)
activation1 = layers.ReLU()(batch_norm1)
pool2 = layers.MaxPooling2D((4, 4))(activation1)

dropout_pool2 = layers.Dropout(0.3)(pool2)

flat1 = layers.Flatten()(dropout_pool2)
attention_weights_text = layers.Dense(units=1, activation='tanh')(flat1)
attention_weights_text = layers.Flatten()(attention_weights_text)
attention_weights_text = layers.Activation('softmax')(attention_weights_text)
attention_weights_text = layers.RepeatVector(256)(attention_weights_text)  # Adjust the value as needed
attention_weights_text = layers.Permute((2, 1))(attention_weights_text)

context_vector_image = layers.Multiply()([flat1, attention_weights_text])

# Define the text input
text_input = tf.keras.Input(shape=(100, ), name='text_input')  # Variable sequence length
embedding = layers.Embedding(input_dim=len(characters)+ 2, output_dim=400)(text_input)  # Adjust vocab_size and output_dim
conv1d = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(embedding)
pooling = layers.MaxPooling1D(pool_size=2)(conv1d)
rnn1 = layers.GRU(256, dropout=0.5)(pooling)
attention_weights_text = layers.Dense(units=1, activation='tanh')(rnn1)
attention_weights_text = layers.Flatten()(attention_weights_text)
attention_weights_text = layers.Activation('softmax')(attention_weights_text)
attention_weights_text = layers.RepeatVector(256)(attention_weights_text)  # Adjust the value as needed
attention_weights_text = layers.Permute((2, 1))(attention_weights_text)

context_vector_text = layers.Multiply()([rnn1, attention_weights_text])

# Define the text-dom input
text_input_bilstm = tf.keras.Input(shape=(2500, ), name='text_input_bilstm')  # Variable sequence length
embedding_bilstm = layers.Embedding(input_dim=len(tags)+2, output_dim=800)(text_input_bilstm)  # Adjust vocab_size and output_dim
conv1d = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(embedding_bilstm)
pooling = layers.MaxPooling1D(pool_size=2)(conv1d)
bi_lstm_bilstm = layers.Bidirectional(layers.LSTM(256))(pooling)
dropout = layers.Dropout(0.4)(bi_lstm_bilstm)

attention_weights_text_bilstm = layers.Dense(units=1, activation='tanh')(dropout)
attention_weights_text_bilstm = layers.Flatten()(attention_weights_text_bilstm)
attention_weights_text_bilstm = layers.Activation('softmax')(attention_weights_text_bilstm)
attention_weights_text_bilstm = layers.RepeatVector(512)(attention_weights_text_bilstm)  # Adjust the value as needed
attention_weights_text_bilstm = layers.Permute((2, 1))(attention_weights_text_bilstm)


context_vector_text_bilstm = layers.Multiply()([bi_lstm_bilstm, attention_weights_text_bilstm])

# # Concatenate the outputs from image and text branches
concatenated = layers.Concatenate()([context_vector_image,context_vector_text,context_vector_text_bilstm])

# Add fully connected layers for further processing
dense1 = layers.Dense(512, activation='relu')(concatenated)
dropo = layers.Dropout(0.5)(dense1)
dense1 = layers.Dense(128, activation='relu')(dropo)
output = layers.Dense(1, activation='sigmoid')(dense1)  # Binary classification output

# Create the model with multiple inputs
model = models.Model(inputs=[image_input,text_input,text_input_bilstm], outputs=output)

# Compile the model
otp = Adam(learning_rate=0.001)
model.compile(optimizer= otp,
              loss='binary_crossentropy',  # Binary classification loss
              metrics=['accuracy'])

# Print the model summary
model.summary()
plot_model(model,show_shapes=True)

# training model
doml_train = np.array(doml_train)
imgl_train = np.array(imgl_train)
strucl_train = np.array(strucl_train)
doml_test = np.array(doml_test)
imgl_test = np.array(imgl_test)
strucl_test = np.array(strucl_test)
labels_train = labels_train.reshape((-1,1))

model.fit(
    [imgl_train,strucl_train,doml_train],
    labels_train,
    epochs=25,
    batch_size=64,
    validation_split=0.2)

# save model
model.save("path_to_save_model")

# evaluate model
test_predict=model.predict([imgl_test,strucl_test,doml_test])
a = []
for i in test_predict:
  if(i>=0.5):
    a.append(1.0)
  else:
    a.append(0.0)

ac = accuracy_score(labels_test, a)
pre = precision_score(labels_test, a)
recall = recall_score(labels_test, a)
f1 = f1_score(labels_test, a)

print('accuracy_score：', ac)
print('precision_score：', pre)
print('recall_score：', recall)
print('f1_score：', f1)