import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, ProjectionHead, MixVPRModel

import pdb


class CPIPModel(nn.Module):
    def __init__(
        self,
        temperature,
        image_embedding,
        location_embedding,
        args
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(args.model_name,args.pretrained, args.trainable, args.freeze_image_encoder, args)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=args.projection_dim, dropout=args.dropout)
        self.location_projection = ProjectionHead(embedding_dim=location_embedding, projection_dim=args.projection_dim, dropout=args.dropout)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image Features
        # shape: torch.Size([8, 3, 810, 1440])
        # device(type='cuda', index=0)
        image_features = self.image_encoder(batch["image"])
        location_features = batch["location"]

        # print("image_faeture: ", image_features)
        # print("location feature: ", location_features)
        
        # Getting Image and Location Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        location_embeddings = self.location_projection(location_features)
        #-----------------------------
        #location_embeddings = encode_text(location_input)

        # print("image emb: ", image_embeddings)
        # print("location emb: ", location_embeddings)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        location_embeddings = F.normalize(location_embeddings, p=2, dim=-1)

        
        # Calculating the Loss
        logits = image_embeddings @ location_embeddings.T / self.temperature
        labels = torch.arange(logits.size(0)).long().to(logits.device)

        
        # TODO: experiment with one way loss
        img_to_text_loss = F.cross_entropy(logits, labels)
        text_to_img_loss = F.cross_entropy(logits.T, labels)

        loss = (img_to_text_loss + text_to_img_loss) / 2

        return loss, logits

