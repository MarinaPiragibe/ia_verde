import pandas as pd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
class Modelo:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def codificar(self):
        # One hot encoding para variáveis categóricas
        self.X = pd.get_dummies(self.X)

        # Codificação dos rótulos do target
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)

    def separar_treino_teste(self, test_size=0.3):
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def iniciar_modelo(self):
        self.codificar()
        self.separar_treino_teste()
        model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(self.X_treino.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(set(self.y)), activation='softmax')
        ])

        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        return model
    
    def treinar_e_avaliar_modelo(self, epocas=30, batch_size=32):
        modelo = self.iniciar_modelo()
        modelo.fit(self.X_treino, self.y_treino, epochs=epocas, batch_size=batch_size, verbose=1)
        # Avaliar
        y_pred = modelo.predict(self.X_teste)
        y_pred_classes = y_pred.argmax(axis=1)

        print("Acuracia:", accuracy_score(self.y_teste, y_pred_classes))
        print("Precisao:", precision_score(self.y_teste, y_pred_classes, average='macro'))
        print("Recall:", recall_score(self.y_teste, y_pred_classes, average='macro'))
        print("F1 Score:", f1_score(self.y_teste, y_pred_classes, average='macro'))
        return