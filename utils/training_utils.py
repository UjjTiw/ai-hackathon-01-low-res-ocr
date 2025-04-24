import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from utils.dataset_utils import idx_to_label


def train_model(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Train Loss: {avg_loss:.4f}")


def evaluate_model(model, loader, device, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    score = f1_score(all_labels, all_preds, average="macro")
    print("F1 Macro Score:", score)

    if criterion is not None:
        avg_loss = total_loss / len(loader)
        print(f"Validation Loss: {avg_loss:.4f}")

    return score


def predict_test(model, loader, device):
    """
    Predict using either a single model or an ensemble of models
    """
    if hasattr(model, '__iter__'):  # Check if model is iterable (list/tuple)
        # It's an ensemble
        return predict_test_ensemble(model, loader, device)
    else:
        # It's a single model
        model.eval()
        results = []

        with torch.no_grad():
            for imgs, filenames in loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = [idx_to_label[p] for p in preds]
                results.extend(zip(filenames, labels))

        return results

def predict_test_ensemble(models, loader, device):
    """
    Ensemble prediction using multiple models
    """
    results = []
    
    with torch.no_grad():
        for imgs, filenames in loader:
            imgs = imgs.to(device)
            
            # Get predictions from all models
            batch_predictions = []
            for model in models:
                model.eval()
                outputs = model(imgs)
                # Store logits, not class predictions
                batch_predictions.append(outputs)
            
            # Average the logits from all models
            ensemble_outputs = torch.mean(torch.stack(batch_predictions), dim=0)
            preds = torch.argmax(ensemble_outputs, dim=1).cpu().numpy()
            
            # Convert to labels
            labels = [idx_to_label[p] for p in preds]
            results.extend(zip(filenames, labels))
    
    return results