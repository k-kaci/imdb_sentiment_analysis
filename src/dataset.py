import torch

class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        IMDBDataset returns one sample of the training or validation data
        :param reviews: this is a numpy array
        :param targets: a vector, numpy array
        """
        self.reviews = reviews
        self.target = targets
    def __len__(self):
        len(self.reviews)
    def __getitem__(self, item):
        review = str(self.reviews[item, :])
        target = self.target[item]
        return {"review": torch.tensor(review, dtype=torch.long),
                "target": torch.tensor(target, dtype=torch.float)}

