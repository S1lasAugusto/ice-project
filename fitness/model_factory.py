import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error

from . import essay_utils

class ModelFactory:
    def __init__(self, embedding_type, model_type, dataset, target):
        self.embedding_type = embedding_type
        self.model_type = model_type
        self.dataset = dataset
        self.target = target
        self.embeddings = None
        self.model = None

    def load_embeddings(self, embeddings_path):
        # Load embeddings based on the type specified
        
        if self.embedding_type == "glove":
            embedding_dict = {}
            with open(f"{embeddings_path}glove.6B.300d.txt", "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vectors = np.asarray(values[1:], "float32")
                    embedding_dict[word] = vectors
            self.embeddings = embedding_dict
        elif self.embedding_type == "fast_text":
            embedding_dict = {}
            with open(f"{embeddings_path}wiki-news-300d-1M.vec", "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vectors = np.asarray(values[1:], "float32")
                    embedding_dict[word] = vectors
            self.embeddings = embedding_dict
        elif self.embedding_type == "word2vec":
            embedding_dict = {}
            with open(f"{embeddings_path}word2vec_model.txt", "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vectors = np.asarray(values[1:], "float32")
                    embedding_dict[word] = vectors
            self.embeddings = embedding_dict
        else:
            raise ValueError("Invalid embedding type.")
        return self.embeddings

    def create_model(self):
        # Create a model based on the type specified
        if self.model_type == "lstm":
            model = tf.keras.models.Sequential()
            model.add(
                tf.keras.layers.LSTM(
                    200,
                    dropout=0.4,
                    recurrent_dropout=0.4,
                    return_sequences=True,
                )
            )
            model.add(tf.keras.layers.LSTM(64, recurrent_dropout=0.4))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(1, activation="relu"))
            model.compile(
                loss="mean_squared_error", optimizer="rmsprop", metrics=["mae"]
            )            
        elif self.model_type == "gru":
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.GRU(128, dropout=0.2, recurrent_dropout=0.2))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(1, activation="relu"))
            model.compile(
                loss="mean_squared_error", optimizer="rmsprop", metrics=["mae"]
            )
        elif self.model_type == "simplernn":
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.SimpleRNN(128, dropout=0.2))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(1, activation="relu"))
            model.compile(
                loss="mean_squared_error", optimizer="rmsprop", metrics=["mae"]
            )
        else:
            raise ValueError("Invalid regressor type.")
        return model

    def train(self, embeddings_path):
        cv = KFold(n_splits=2, shuffle=True)
        results = []
        mse_scores = []

        count = 1
        for traincv, testcv in cv.split(self.dataset):
            print("\n--------Fold {}--------\n".format(count))
            test_x, train_x, y_test, y_train = (
                self.dataset.iloc[testcv],
                self.dataset.iloc[traincv],
                self.target.iloc[testcv],
                self.target.iloc[traincv],
            )

            train_essays = train_x["essay"]
            test_essays = test_x["essay"]

            self.embeddings = self.load_embeddings(embeddings_path)

            num_features = 300

            # Generate training and testing data word vectors.
            clean_train_essays = []
            for essay_v in train_essays:
                clean_train_essays.append(
                    essay_utils.essay_to_wordlist(essay_v, remove_stopwords=True)
                )

            train_data_vecs = essay_utils.get_avg_feature_vecs(
                clean_train_essays, self.embeddings, num_features
            )

            clean_test_essays = []
            for essay_v in test_essays:
                clean_test_essays.append(
                    essay_utils.essay_to_wordlist(essay_v, remove_stopwords=True)
                )

            test_data_vecs = essay_utils.get_avg_feature_vecs(
                clean_test_essays, self.embeddings, num_features
            )

            train_data_vecs = np.array(train_data_vecs)
            test_data_vecs = np.array(test_data_vecs)

            # Reshaping train and test vectors to  3 dimensions. (1 represents one timestep)
            train_data_vecs = np.reshape(
                train_data_vecs, (train_data_vecs.shape[0], 1, train_data_vecs.shape[1])
            )
            test_data_vecs = np.reshape(
                test_data_vecs, (test_data_vecs.shape[0], 1, test_data_vecs.shape[1])
            )

            self.model = self.create_model()
            self.model.fit(train_data_vecs, y_train, batch_size=64, epochs=50)
            y_pred = self.model.predict(test_data_vecs)

            # Calculate the MSE for the current fold
            mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)

            # Round y_pred to the nearest integer.
            y_pred = np.round(y_pred)

            # Evaluate the model on the evaluation metric. "Quadratic mean averaged Kappa"
            result = cohen_kappa_score(y_test.values, y_pred, weights="quadratic")
            print("Kappa Score: {}".format(result))
            avg_mse = np.mean(mse_scores)
            results.append(result)

            count += 1
            self.model.summary()
            return self.model_type, avg_mse