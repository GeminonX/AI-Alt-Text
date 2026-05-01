import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor
import torch.nn as nn
import torch.optim as optim

from dataset import ScienceQADataset
from model import SubjectClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BACKBONE_NAME = "google/vit-base-patch16-224"
NUM_CLASSES = 8
BATCH_SIZE = 2
EPOCHS = 1
LR = 1e-4

def collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    processed = processor(images=images, return_tensors="pt")
    processed["labels"] = labels
    return processed

def main():
    print("Loading processor...")
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    print("Creating dataset...")
    dataset = ScienceQADataset()
    print("Dataset length:", len(dataset))
    print("Labels:", dataset.label2id)

    print("Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    print("Creating model...")
    model = SubjectClassifier(BACKBONE_NAME,num_labels=len(dataset.label2id))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    print("Starting training...")
    for epoch in range(EPOCHS):
        print("Starting epoch", epoch + 1)

        for i, batch in enumerate(dataloader):
            print("Batch", i)
            
            pixel_values = batch["pixel_values"]
            labels = batch["labels"]

            optimizer.zero_grad()
            logits = model(pixel_values)  # Forward pass
            loss = criterion(logits, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()

            print("Loss: ", loss.item())
         
    """
     processor = AutoImageProcessor.from_pretrained(BACKBONE_NAME)

    dataset = ScienceQADataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )

    model = SubjectClassifier(BACKBONE_NAME, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch in loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
        """

def predict_batch(model, batch, device):
    model.eval()
    with torch.no_grad():
        logits = model(batch["pixel_values"].to(device))
        preds = torch.argmax(logits, dim=1)
    return preds.cpu()


if __name__ == "__main__":
    main()