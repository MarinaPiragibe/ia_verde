import torch
from module.encoders import RecordEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from module.encoders import NGramEncoder, RecordEncoder


class EncodersModel():
    def __init__(self, model):
        self.model = model

    def start_encoders(self, X, dimension, num_levels, low, high):
        self.record_encoder = RecordEncoder(
            out_features=dimension,
            size=X.shape[1], 
            levels=num_levels,
            low=low,
            high=high
        )

        self.ngram_encoder = NGramEncoder(
            out_features=dimension,
            levels=num_levels,
            low=low,
            high=high
        )


    def run_encoders(self, X, y, device, oper):

        with torch.no_grad():
            print(X.dtypes)

            samples = torch.tensor(X.values).to(device)
            self.labels = torch.tensor(y).to(device)

            # RecordEncoder
            self.X_record_encoder = self.record_encoder(samples.detach().clone())

            # NGramEncoder
            self.X_ngram_encoder = self.ngram_encoder(samples.detach().clone(), oper=oper)

    def split_train_test(self):
        self.X_train_ngram, self.X_test_ngram, self.y_train_ngram, self.y_test_ngram = train_test_split(self.X_ngram_encoder, self.labels, test_size=0.3, random_state = 0) 
        self.X_train_record, self.X_test_record, self.y_train_record, self.y_test_record = train_test_split(self.X_record_encoder, self.labels, test_size=0.3, random_state = 0) 


    def run_all_types(self):
        with torch.no_grad():

            self.model.fit(self.X_train_record,self.y_train_record)
            predictions = self.model.predict(self.X_test_record.to(torch.int8))  
            acc = accuracy_score(predictions, self.y_test_record)
            print("BinHD Record Encoder: Accuracy = ", acc)

            self.model.fit(self.X_train_ngram,self.y_train_ngram)
            predictions = self.model.predict(self.X_test_ngram.to(torch.int8))  
            acc = accuracy_score(predictions, self.y_test_ngram)
            print("BinHD Ngram Encoder: Accuracy = ", acc)
