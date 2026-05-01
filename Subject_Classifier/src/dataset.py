from torch.utils.data import Dataset
from datasets import load_dataset

class ScienceQADataset(Dataset):
    def __init__(self, split="validation", transform=None):
        self.dataset = load_dataset("lmms-lab/ScienceQA", "ScienceQA-IMG", split=split)

        # filter out any weird edge cases (safety)
        self.dataset = self.dataset.filter(lambda x: x["image"] is not None)

        # build label mapping
        labels = sorted(set(self.dataset["subject"]))
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"].convert("RGB")
        label = self.label2id[sample["subject"]]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label
        }