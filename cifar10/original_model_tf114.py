from tensorflow.python.keras.datasets import cifar10 # import cifar10
from tensorflow.python.keras.utils import to_categorical # カテゴリ化
from tensorflow.python.keras.models import Sequential # 畳み込み層作成
from tensorflow.python.keras.layers import Conv2D # 畳み込み層追加
from tensorflow.python.keras.layers import MaxPooling2D # プーリング層追加
from tensorflow.python.keras.layers import Dropout # ドロップアウトレイヤー追加
from tensorflow.python.keras.layers import Flatten # Flattenレイヤーの追加
from tensorflow.python.keras.layers import Dense # 全結合層の追加
from tensorflow.python.keras.callbacks import TensorBoard # tensorboard

# データインポート
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# データの整形
# データ次元確認
print('x_train.shape:', x_train.shape)
print('x_test.shape:', x_test.shape)
print('y_train.shape:', y_train.shape)
print('y_test.shape:', y_test.shape)

# 正規化
x_train = x_train/255.
x_test = x_test/255.

# one hot ベクトル化
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 畳み込み層構築
model = Sequential()

# 畳み込み層追加
model.add(Conv2D(filters=32, input_shape=(32,32,3),
                kernel_size=(3,3),strides=(1,1),padding='same', # padding='same'でゼロパディング
                activation='relu'))

model.add(Conv2D(filters=32, kernel_size=(3,3),strides=(1,1),
                padding='same',activation='relu'))

# プーリング層追加
model.add(MaxPooling2D(pool_size=(2,2)))

# ドロップアウトレイヤー追加
model.add(Dropout(0.25))

# 畳み込み層とプーリング層追加
model.add(Conv2D(filters=64, input_shape=(32,32,3),
                kernel_size=(3,3),strides=(1,1),padding='same', # padding='same'でゼロパディング
                activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3),strides=(1,1),
                padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 全結合層の追加
# プーリング層追加後のモデルの出力形式
print(model.output_shape) # => (None, 8, 8, 64)

# Flattenレイヤーの追加
model.add(Flatten())
print(model.output_shape) # => (None, 4096)

# 全結合層の追加
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

# モデル学習
model.compile(optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy'])
tsb = TensorBoard(log_dir='./logs')
history_model1 = model.fit(x_train,
                          y_train,
                          batch_size=32,
                          epochs=20,
                          validation_split=0.2,
                          callbacks=[tsb])

# モデル構造
print(model.summary())

# スコア算出
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.save('original_model_tf114_cifar10.h5')


