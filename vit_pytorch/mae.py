import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange

from vit_pytorch.vit import Transformer

# import lpips
from torch.autograd import Variable

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        apply_decoder_pos_emb_all = False # whether to (re)apply decoder positional embedding to encoder unmasked tokens
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.apply_decoder_pos_emb_all = apply_decoder_pos_emb_all

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        # # perceptual loss
        # self.perc_loss = lpips.LPIPS(net='alex', pnet_rand=True)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens_all = self.patch_to_emb(patches)
        tokens_all = tokens_all + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens_all[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]
        # print(torch.max(masked_patches))
        weight_patches = (torch.sum(torch.abs(masked_patches), dim=2) / masked_patches.size(2)).unsqueeze(-1).contiguous()

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens, if desired

        if self.apply_decoder_pos_emb_all:
            decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[:, :num_masked]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # safe out the reconstructed images
        # print(decoded_tokens[:, :num_masked].size())
        # print(tokens_all[batch_range, unmasked_indices].size())
        ordered_tokens = torch.zeros(decoded_tokens.size(), device = device)
        # ordered_tokens[batch_range, rand_indices] = decoded_tokens
        ordered_tokens[batch_range, rand_indices[:, :num_masked]] = decoded_tokens[:, :num_masked]
        ordered_tokens[batch_range, rand_indices[:, num_masked:]] = self.enc_to_dec(tokens)
        reconstructed_pixels = self.to_pixels(ordered_tokens)
        # print(reconstructed_pixels.size())
        p1, p2 = self.encoder.patch_size, self.encoder.patch_size
        h, w = int(self.encoder.image_size/self.encoder.patch_size), int(self.encoder.image_size/self.encoder.patch_size)
        # print(h, w, p1, p2)
        reverse_patch = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = h, w = w, p1 = p1, p2 = p2),
        )
        reconstructed_pixels = reverse_patch(reconstructed_pixels)

        # calculate reconstruction loss

        # # option - 1: perceptual
        # patches_ = Variable(reverse_patch(patches), requires_grad=True)
        # reconstructed_pixels_ = (reconstructed_pixels - torch.min(reconstructed_pixels)) / (torch.max(reconstructed_pixels) - torch.min(reconstructed_pixels))
        # patches_ = (patches_ - torch.min(patches_)) / (torch.max(patches_) - torch.min(patches_))
        # perc_loss = torch.mean(self.perc_loss(reconstructed_pixels_, patches_, normalize=True))

        # option - 2:  mse
        # mse_loss = F.mse_loss(weight_patches*pred_pixel_values, weight_patches*masked_patches)
        mse_loss = F.mse_loss(pred_pixel_values,masked_patches)

        recon_loss = mse_loss

        return recon_loss, reconstructed_pixels
