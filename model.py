import torch
import torch.nn as nn
from PIL import Image
import pickle
import torchvision
from torchvision import transforms
import torchvision.models as models
import os


path = os.path.dirname(os.path.abspath("model.py"))

class VQA_baseline(nn.Module):
    def __init__(self):
        super(VQA_baseline, self).__init__()

        resnet = models.resnet18(pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

        imsize = (256, 256)
        self.loader = transforms.Compose([
            transforms.Resize(imsize),
            transforms.ToTensor()])

        with open(path + "/static/model/q_word_to_ix.txt", "rb") as inp:
            self.q_word_to_ix = pickle.load(inp)

        with open(path + "/static/model/a_to_ix.txt", "rb") as inp:
            self.a_to_ix = pickle.load(inp)

        self.ix_to_a = dict((v, k) for k, v in self.a_to_ix.items())

        self.words_features = nn.Linear(len(self.q_word_to_ix), 512)
        self.image_features = feature_extractor
        self.classifier = nn.Sequential(nn.Linear(1024, 300),
                                        nn.ReLU(),
                                        nn.Linear(300, len(self.a_to_ix)))


    def to_onehot(self, vector, vocab):
        x = torch.zeros(len(vocab))
        for i in vector:
            if i in vocab:
                x[vocab[i]] = 1
        return x


    def forward(self, question, image_path):
        image = Image.open(image_path)
        image = self.loader(image).unsqueeze(0)
        question = self.to_onehot(question.split(), self.q_word_to_ix)
        x1 = self.image_features(image)
        x2 = self.words_features(question)
        x = torch.cat([x1.view(1, -1), 10 * x2.view(1, -1)], 1)
        answer = self.classifier(x)
        prediction = torch.squeeze(answer)
        l1, index1 = torch.max(prediction, -1)
        result = self.ix_to_a[index1.item()]
        return result