class EarlyStopping:
    def __init__(self, patience=3, mode='max'):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.mode = mode

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return

        improve = (current_score > self.best_score) if self.mode == 'max' else (current_score < self.best_score)

        if improve:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
