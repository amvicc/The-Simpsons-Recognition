import os

import torch.nn as nn
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from pathlib import Path


class SimpsonsDataset(Dataset):
    def __init__(self, files, mode, augmentations):
        super().__init__()
        self.files = files
        self.mode = mode
        self.augmentations = augmentations

        self.len_ = len(self.files)
        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump:
                pickle.dump(self.label_encoder, le_dump)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        x = self.load_sample(self.files[index])

        if self.mode == 'aug':
            x = self.augmentations(x)

        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype='float32')

        x = transform(x)

        if self.mode == 'test':
            return x
        else:

            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

    def _prepare_sample(self, image):
        image = image.resize((224, 224))

        return np.array(image)


class Core:
    model = None
    device = None
    label_encoder = None

    def __init__(self, model):
        self.model = model
        self.label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
        self.device = torch.device("cuda")

    def predict(self, aug_loader):
        with torch.no_grad():
            logits = []

            for inputs in aug_loader:
                inputs = inputs.to(self.device)
                self.model.eval()
                outputs = self.model(inputs).cpu()
                logits.append(outputs)

        probs = torch.nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
        return probs

    @staticmethod
    def save_file(file):
        filename = secure_filename(file.filename)
        destination = "/files/" + filename
        file.save(destination)

    def predict_simpson_name(self):
        # self.save_file(file)
        files = Path('C:\\Users\\Anton\\dev\\The-Simpsons-Recognition\\Application\\files')
        test_files = sorted(list(files.rglob('*.jpg')))
        test_dataset = SimpsonsDataset(test_files, mode="test", augmentations=None)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=42)
        probs = self.predict(test_loader)
        preds = self.label_encoder.inverse_transform(np.argmax(probs, axis=1))
        test_filenames = [path.name for path in test_dataset.files]
        submit = pd.DataFrame({'Id': test_filenames, 'Expected': preds})
        submit.head()
        f = open("result.txt", "w+")
        f.write(preds[0])
        f.close()


class SimpleModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=160, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(0.8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.f1 = nn.Sequential(
            nn.Linear(7 * 7 * 160, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.f2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.f3 = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = x.view(-1, 7 * 7 * 160)
        x = self.f1(x)
        x = self.f2(x)

        logits = self.f3(x)
        return logits


if __name__ == '__main__':
    print(os.path.abspath(os.getcwd()))
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        core = Core(model)
        core.predict_simpson_name()
