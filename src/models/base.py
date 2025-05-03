from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train_model(self, train_texts, train_labels, val_texts, val_labels, config, log_callback=False):
        pass

    @abstractmethod
    def predict(self, texts):
        pass

    @abstractmethod
    def save_state(self, path):
        pass

    @abstractmethod
    def evaluate_model(self, texts, labels):
        pass
