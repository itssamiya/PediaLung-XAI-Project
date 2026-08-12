class EarlyStopping:

    def __init__(self, patience=10):

        self.patience = patience
        self.best = -float("inf")
        self.counter = 0

    def step(self, macro_f1):

        if macro_f1 > self.best:

            self.best = macro_f1
            self.counter = 0
            return False

        self.counter += 1

        return self.counter >= self.patience
