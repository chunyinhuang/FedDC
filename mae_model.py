import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.vit import ViT
from vit_pytorch.mae import MAE

random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

# v = ViT(
#     image_size = 96,
#     patch_size = 8,
#     num_classes = 1000,
#     dim = 1024,
#     depth = 6,
#     heads = 8,
#     mlp_dim = 2048
# )

# mae = MAE(
#     encoder = v,
#     masking_ratio = 0.4,   # the paper recommended 75% masked patches
#     decoder_dim = 512,      # paper showed good results with just 512
#     decoder_depth = 6       # anywhere from 1 to 8
# )


def Get_MAE(image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024, # changed for covidx
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
    ):

    v = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim
    )

    mae = MAE(
        encoder = v,
        decoder_dim = decoder_dim,      # paper showed good results with just 512
        masking_ratio = masking_ratio,   # the paper recommended 75% masked patches
        decoder_depth = decoder_depth,       # anywhere from 1 to 8
        apply_decoder_pos_emb_all = True
    )

    return mae


class MAE_encoder(nn.Module):
    def __init__(
        self,
        encoder,
        dim=1024
    ):
        super().__init__()

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        
        self.to_latent = nn.Identity()


    def forward(self, imgs):

        # get patches

        patches = self.to_patch(imgs)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)
        encoded_tokens = encoded_tokens.mean(dim = 1)
        encoded_tokens = self.to_latent(encoded_tokens)

        
        return encoded_tokens

