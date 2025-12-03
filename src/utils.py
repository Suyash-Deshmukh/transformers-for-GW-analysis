class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_auroc_score = 0

    def early_stop(self, auroc_score):
        if auroc_score > self.max_auroc_score:
            self.max_auroc_score = auroc_score
            self.counter = 0
        elif auroc_score < (self.max_auroc_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
