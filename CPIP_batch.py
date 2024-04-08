import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import cv2

import argparse
import pickle
from tqdm import tqdm
import json
import os
import csv
import re
import glob
import pdb

import pandas as pd
import torch
from CPIP import CPIPModel
from dataset import CPIPDataset, get_transforms, Dataset_for_query
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils_file import AvgMeter, get_lr


def parsing_argument():
    parser = argparse.ArgumentParser()

    # !! Remember to change these values !!
    parser.add_argument('--process_data', type= int, default = 1, help = "Whether to process the data: 1: yes; 0: no")
    parser.add_argument('--do_train', type= int, default = 1, help = "Whether to train the model: 1: yes; 0: no")
    parser.add_argument('--generate_query', type= int, default = 0, help = "Whether to generate query images: 1: yes; 0: no")
    parser.add_argument('--experiment_name', type=str, default="0405_VPR", help="Experiment name")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--model_name', type=str, default='MixVPR', help="Model name")
    parser.add_argument('--dataset', type=str, default='Tokyo247', help="Dataset name")
    # !!                                 !!

    parser.add_argument('--vpr_threshold', type=float, default=0.05)
    parser.add_argument('--resolution_decay', type= int, default = 4, help = "Resolution decay factor")
    parser.add_argument('--image_path', type=str, default="/scratch/jh7956/Datasets/Tokyo247", help="Path to the data")
    parser.add_argument('--vector_path', type=str, default="/scratch/jh7956/Datasets/Tokyo247/vector", help="Path to the vector")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay")
    parser.add_argument('--patience', type=int, default=2, help="Patience")
    parser.add_argument('--factor', type=float, default=0.5, help="Factor")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument('--image_embedding', type=int, default=2048, help="Image embedding size")
    parser.add_argument('--location_embedding', type=int, default=5, help="Location embedding size")
    parser.add_argument('--pretrained', type=bool, default=True, help="Use pretrained model")
    parser.add_argument('--trainable', type=bool, default=True, help="Model is trainable")
    parser.add_argument('--freeze_image_encoder', type=bool, default=False, help="Freeze image encoder")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature")
    parser.add_argument('--image_width', type=int, default=640, help="Image width")
    parser.add_argument('--image_height', type=int, default=380, help="Image height")
    parser.add_argument('--num_projection_layers', type=int, default=1, help="Number of projection layers")
    parser.add_argument('--projection_dim', type=int, default=256, help="Projection dimension")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--train_test_split', type=float, default=0.05, help="Train test split")

    args = parser.parse_args()
    return args

# ----------------- CLIP -----------------
def prepare_data(data_path, test_size=0.0, random_state=42, batch_size=100):
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(".jpg")])
    
    # 计算批次数量
    total_images = len(image_files)
    num_batches = (total_images + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min(start_index + batch_size, total_images)
        batch_files = image_files[start_index:end_index]

        data = []
        for image_file in batch_files:
            location = parsing_database_image(image_file)
            data.append({"image": image_file, "location": location})

        df = pd.DataFrame(data)

        # 在每个批次中分割数据
        train_df, valid_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)

        yield train_df, valid_df


def prepare_data_for_query(data_path, batch_size=100):
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(".jpg")])

    total_images = len(image_files)
    num_batches = (total_images + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min(start_index + batch_size, total_images)
        batch_files = image_files[start_index:end_index]

        data = [{"image": image_file} for image_file in batch_files]
        df = pd.DataFrame(data)

        yield df


def build_loaders(dataframe, mode, image_path, args):
    transforms = get_transforms(mode=mode, args=args)
    dataset = CPIPDataset(dataframe, image_path, transforms=transforms, args=args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= args.batch_size,
        num_workers=args.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def build_loaders_for_query(dataframe, mode, image_path, args):
    transforms = get_transforms(mode=mode, args=args)
    dataset = Dataset_for_query(dataframe, image_path, transforms=transforms, args=args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, args):
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    # image input size: torch.Size([3, 518, 518])

    for batch in tqdm_object:
        batch = {k: v.to(args.device) for k, v in batch.items() if k != "caption"}
        optimizer.zero_grad()
        loss, logits = model(batch)# seg fault here
        labels = torch.arange(logits.size(0)).long().to(logits.device)
        accuracy = calculate_accuracy(logits, labels)

        loss.backward()
        optimizer.step()

        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        accuracy_meter.update(accuracy, count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, train_accuracy=accuracy_meter.avg, lr=get_lr(optimizer))

    return loss_meter.avg, accuracy_meter.avg

def valid_epoch(model, valid_loader, args):
    model.eval()
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    with torch.no_grad():
        for batch in tqdm_object:
            batch = {k: v.to(args.device) for k, v in batch.items() if k != "caption"}
            loss, logits = model(batch)
            labels = torch.arange(logits.size(0)).long().to(logits.device)
            accuracy = calculate_accuracy(logits, labels)

            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            accuracy_meter.update(accuracy, count)

            tqdm_object.set_postfix(valid_loss=loss_meter.avg, valid_accuracy=accuracy_meter.avg)

    return loss_meter.avg, accuracy_meter.avg

def calculate_accuracy(logits, labels):
    _, predicted = logits.max(1)
    correct = predicted.eq(labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy
# ---------------------------------------


# ----------------- VPR -----------------
def compute_topk_match_vector(query_vectors, database_vectors, k=1, batch_size=100) -> np.ndarray:
    query_torch = torch.tensor(query_vectors).float().cuda()
    database_torch = torch.tensor(database_vectors).float().cuda()
    
    num_queries = query_vectors.shape[0]
    matched_indices = torch.zeros((num_queries, k), dtype=torch.int64).cuda()

    for start_idx in tqdm(range(0, num_queries, batch_size), desc="Computing matches"):
        end_idx = min(start_idx + batch_size, num_queries)
        batch = query_torch[start_idx:end_idx]

        prod = torch.einsum('ik,jk->ij', batch, database_torch)

        # Find the Top-K matches in the database for each vector in the batch
        _, indices = torch.topk(prod, k, dim=1, largest=True)
        matched_indices[start_idx:end_idx, :] = indices

    return matched_indices.cpu().numpy()

def vpr_recall(location: np.ndarray, matched_location: np.ndarray, threshold: int) -> bool:
    """
    Simulates the VPR recall mechanism using precomputed matched indices.
    
    Args:
        matched_index (int): The index of the matched image.
        location (np.ndarray): The location associated with the descriptor.
        
    Returns:
        bool: Whether VPR recall was successful.
    """
    if np.linalg.norm(matched_location[:2] - location[:2]) <= threshold:
        return True
    else:
        return False

def do_vpr(database_vectors, query_vectors, args):
    #top_k_values = [1, 2, 5, 10, 20]
    top_k_values = [1]
    vpr_success_rates = {}

    for k in top_k_values:
        print(f"Computing VPR for Top-{k}")
        matched_indices = compute_topk_match_vector(query_vectors, database_vectors, k=k)
        
        vpr_success = np.zeros((len(query_vectors), k), dtype=np.bool_)

        for index in tqdm(range(len(query_vectors)), total=len(query_vectors), desc=f'Examining VPR for Top-{k}'):
            for ki in range(k):  
                vpr_success[index, ki] = vpr_recall(query_vectors[index], database_vectors[matched_indices[index, ki]], args.vpr_threshold)
        
        vpr_success_rate = np.any(vpr_success, axis=1).mean()
        vpr_success_rates[f'Top-{k}'] = vpr_success_rate

        print(f"{k} VPR successful rate: {vpr_success_rate*100}%")

    return vpr_success_rates


    """
    Get the image features from the model and store it to the vector_path. Used for both database and query images.

    Args:
        model: The model to extract features from.
        image_path (str): Path to the input image.
        output_path (str): Directory to save the features.
    """

    model.eval()
    features = []

    with torch.no_grad(): 
        for batch in data_loader:
            # Assuming 'image' key in batch dict
            image_data = batch['image'].to(device)
            image_features = model.image_encoder(image_data)
            
            # Optionally, convert features to another form, e.g., numpy array
            image_features_np = image_features.cpu().numpy()
            features.append(image_features_np)

    # Save features to file
    features_array = np.concatenate(features, axis=0)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_path, features_array)
# ---------------------------------------

# ----------------- Dataset -----------------
def parsing_database_image(filename):
    parts = filename.split('@')
    try:
        # split with @
        parts = parts[:9]
        # discard index 2 and 3
        parts[-1]=parts[-1].split('.jpg')[0]
        feature = [float(parts[1]), float(parts[2]), float(parts[5]), float(parts[6]), float(parts[8])]
    
    except (ValueError, AssertionError):
            # If conversion to float fails, skip this file
            print(f"Skipping file due to invalid format: {filename}")

    return np.array(feature)

def extract_features_from_images(model, image_path, output_path, data_loader, device):
    """
    Get the image features from the model and store it to the vector_path. Used for both database and query images.

    Args:
        model: The model to extract features from.
        image_path (str): Path to the input image.
        output_path (str): Directory to save the features.
    """
    model.eval()
    features = []

    with torch.no_grad(): 
        for batch in data_loader:
            # Assuming 'image' key in batch dict
            image_data = batch['image'].to(device)
            image_features = model.image_encoder(image_data)
            
            # Optionally, convert features to another form, e.g., numpy array
            image_features_np = image_features.cpu().numpy()
            features.append(image_features_np)

    # Save features to file
    features_array = np.concatenate(features, axis=0)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_path, features_array)
# ---------------------------------------

def main():
    print("Start the pipeline...")
    args = parsing_argument()
    model_pt = args.model_name + args.dataset + ".pt"
    image_database_path = os.path.join(args.image_path, "database", "raw")
    image_query_path = os.path.join(args.image_path, "queries", "raw")
    model = CPIPModel(args.temperature, args.image_embedding, args.location_embedding, args).to(args.device)
    print(model)
    if os.path.exists(model_pt):
        model.load_state_dict(torch.load(model_pt))
        print("Loaded Best Model!")
    
    # 1. Train the model
    if args.do_train == 1:
        print("Start training the model...")  
        best_loss = float("inf")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=args.patience, factor=args.factor)
        step = "epoch"

        for train_df, valid_df in prepare_data(image_database_path, test_size=args.train_test_split, batch_size=args.batch_size):
            train_loader = build_loaders(train_df, mode="train", image_path=image_database_path, args=args)
            valid_loader = build_loaders(valid_df, mode="valid", image_path=image_database_path, args=args)
            
            with open('metrics.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"])

                for epoch in range(args.epochs):
                    print(f"Epoch: {epoch + 1}")
                    model.train()
                    train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, lr_scheduler, step, args)
                    model.eval()
                    with torch.no_grad():
                        valid_loss, valid_accuracy = valid_epoch(model, valid_loader, args=args)

                    writer.writerow([epoch + 1, train_loss, valid_loss, train_accuracy, valid_accuracy])
                    print(f"Train Loss: {train_loss}, Valid Loss: {valid_loss}, Train Accuracy: {train_accuracy}, Valid Accuracy: {valid_accuracy}")

                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        torch.save(model.state_dict(), args.experiment_name + ".pt")
                        print("Saved Best Model!")


    # 2. Store the query and database vectors
    database_vector_path = os.path.join(args.vector_path, "database_vectors.npy")
    query_vector_path = os.path.join(args.vector_path, "query_vectors.npy")

    if args.process_data == 1:
        print("Start processing the data...")

        all_df = pd.concat([train_df, valid_df])
        all_loader = build_loaders(all_df, mode="valid", image_path =image_database_path, args=args)

        # Extract features from the database images and store them
        if not os.path.exists(database_vector_path):
            extract_features_from_images(model, image_database_path, database_vector_path, all_loader, args.device)
            print("Database vectors stored")

    
    # 3. Load the database vectors and query vectors
    database_vectors = np.load(database_vector_path)

    print(f"Database vectors: {database_vectors.shape}")

    # 5. Use the model to do VPR on the validation set
    vpr_success_rates = do_vpr(database_vectors, database_vectors, args=args)




if __name__=="__main__":
    main()
    