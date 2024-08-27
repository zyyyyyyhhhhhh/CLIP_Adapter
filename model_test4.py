from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj


class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


# class CLIPVAD(nn.Module):
#     def __init__(self,
#                  num_class: int,
#                  embed_dim: int,
#                  visual_length: int,
#                  visual_width: int,
#                  visual_head: int,
#                  visual_layers: int,
#                  attn_window: int,
#                  prompt_prefix: int,
#                  prompt_postfix: int,
#                  label_map:dict,
#                  device):
#         super().__init__()
#
#         self.num_class = num_class
#         self.visual_length = visual_length
#         self.visual_width = visual_width
#         self.embed_dim = embed_dim
#         self.attn_window = attn_window
#         self.prompt_prefix = prompt_prefix
#         self.prompt_postfix = prompt_postfix
#         self.device = device
#
#         self.temporal = Transformer(
#             width=visual_width,
#             layers=visual_layers,
#             heads=visual_head,
#             attn_mask=self.build_attention_mask(self.attn_window)
#         )
#
#         width = int(visual_width / 2)
#         self.gc1 = GraphConvolution(visual_width, width, residual=True)
#         self.gc2 = GraphConvolution(width, width, residual=True)
#         self.gc3 = GraphConvolution(visual_width, width, residual=True)
#         self.gc4 = GraphConvolution(width, width, residual=True)
#         self.disAdj = DistanceAdj()
#         self.linear = nn.Linear(visual_width, visual_width)
#         self.gelu = QuickGELU()
#
#         self.mlp1 = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(visual_width, visual_width * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(visual_width * 4, visual_width))
#         ]))
#         self.mlp2 = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(visual_width, visual_width * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(visual_width * 4, visual_width))
#         ]))
#         self.classifier = nn.Linear(visual_width, 1)
#
#         self.clipmodel, _ = clip.load("ViT-B/16", device)
#         for clip_param in self.clipmodel.parameters():
#             clip_param.requires_grad = False
#
#         self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
#         self.text_prompt_embeddings = nn.Embedding(77, embed_dim)
#
#         self.initialize_parameters()
#
#         # 初始化 PromptLearner
#         self.prompt_learner = PromptLearner(classnames=list(label_map.values()), clip_model=self.clipmodel, n_ctx=4, device = self.device)
#
#     def initialize_parameters(self):
#         nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
#         nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)
#
#     def build_attention_mask(self, attn_window):
#         # lazily create causal attention mask, with full attention between the vision tokens
#         # pytorch uses additive attention mask; fill with -inf
#         mask = torch.empty(self.visual_length, self.visual_length)
#         mask.fill_(float('-inf'))
#         for i in range(int(self.visual_length / attn_window)):
#             if (i + 1) * attn_window < self.visual_length:
#                 mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
#             else:
#                 mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0
#
#         return mask
#
#     def adj4(self, x, seq_len):
#         soft = nn.Softmax(1)
#         x2 = x.matmul(x.permute(0, 2, 1))  # B*T*T
#         x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
#         x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
#         x2 = x2 / (x_norm_x + 1e-20)
#         output = torch.zeros_like(x2)
#         if seq_len is None:
#             for i in range(x.shape[0]):
#                 tmp = x2[i]
#                 adj2 = tmp
#                 adj2 = F.threshold(adj2, 0.7, 0)
#                 adj2 = soft(adj2)
#                 output[i] = adj2
#         else:
#             for i in range(len(seq_len)):
#                 tmp = x2[i, :seq_len[i], :seq_len[i]]
#                 adj2 = tmp
#                 adj2 = F.threshold(adj2, 0.7, 0)
#                 adj2 = soft(adj2)
#                 output[i, :seq_len[i], :seq_len[i]] = adj2
#
#         return output
#
#     def encode_video(self, images, padding_mask, lengths):
#         images = images.to(torch.float)
#         position_ids = torch.arange(self.visual_length, device=self.device)
#         position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
#         frame_position_embeddings = self.frame_position_embeddings(position_ids)
#         frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
#         images = images.permute(1, 0, 2) + frame_position_embeddings
#
#         x, _ = self.temporal((images, None))
#         x = x.permute(1, 0, 2)
#
#         adj = self.adj4(x, lengths)
#         disadj = self.disAdj(x.shape[0], x.shape[1])
#         x1_h = self.gelu(self.gc1(x, adj))
#         x2_h = self.gelu(self.gc3(x, disadj))
#
#         x1 = self.gelu(self.gc2(x1_h, adj))
#         x2 = self.gelu(self.gc4(x2_h, disadj))
#
#         x = torch.cat((x1, x2), 2)
#         x = self.linear(x)
#
#         return x
#
#     def encode_textprompt(self, text):
#         prompts = self.prompt_learner()  # 生成 prompts
#         # print(prompts)
#         text_features = self.clipmodel.encode_text(prompts, self.prompt_learner.tokenized_prompts)
#
#         return text_features
#     # def encode_textprompt(self, text):
#     #     # text是prompt语句
#     #     word_tokens = clip.tokenize(text).to(self.device)
#     #     word_embedding = self.clipmodel.encode_token(word_tokens)
#     #     text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat(
#     #         [len(text), 1, 1])
#     #     text_tokens = torch.zeros(len(text), 77).to(self.device)
#     #
#     #     for i in range(len(text)):
#     #         ind = torch.argmax(word_tokens[i], -1)
#     #         text_embeddings[i, 0] = word_embedding[i, 0]
#     #         text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
#     #         text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
#     #         text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]
#     #
#     #     text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
#     #
#     #     return text_features
#
#     def forward(self, visual, padding_mask, text, lengths):
#         # print(text)
#         visual_features = self.encode_video(visual, padding_mask, lengths)
#         logits1 = self.classifier(visual_features + self.mlp2(visual_features))
#
#         text_features_ori = self.encode_textprompt(text)
#         # print(f'text_features_ori.shape:{text_features_ori}')
#         # print(text_features_ori.shape)
#         # print(text_features_ori)
#
#         text_features = text_features_ori
#         logits_attn = logits1.permute(0, 2, 1)
#         visual_attn = logits_attn @ visual_features
#         visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
#         visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
#         text_features = text_features_ori.unsqueeze(0)
#         text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
#         text_features = text_features + visual_attn
#         text_features = text_features + self.mlp1(text_features)
#
#         visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
#         text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
#         text_features_norm = text_features_norm.permute(0, 2, 1)
#         logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07
#
#         return text_features_ori, logits1, logits2

class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 label_map: dict,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, embed_dim)

        # 初始化 PromptLearner
        self.prompt_learner = PromptLearner(classnames=list(label_map.values()), clip_model=self.clipmodel, n_ctx=4, device=self.device)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)
        nn.init.normal_(self.prompt_learner.meta_net.linear1.weight, std=0.02)
        nn.init.zeros_(self.prompt_learner.meta_net.linear1.bias)
        nn.init.normal_(self.prompt_learner.meta_net.linear2.weight, std=0.02)
        nn.init.zeros_(self.prompt_learner.meta_net.linear2.bias)
        nn.init.normal_(self.prompt_learner.ctx, std=0.02)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))  # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def encode_textprompt(self, image_features):
        prompts = self.prompt_learner(image_features)  # 生成基于视频特征的 prompts
        # 展平 prompts 以适应文本编码器
        # batch_size, n_cls, seq_len, dim = prompts.shape
        # prompts = prompts.view(batch_size * n_cls, seq_len, dim)
        # tokenized_prompts = self.prompt_learner.tokenized_prompts.repeat(batch_size, 1)

        text_features = self.clipmodel.encode_text(prompts, self.prompt_learner.tokenized_prompts)

        return text_features  # 恢复形状


    def forward(self, visual, padding_mask, text, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        # print('-----------------------------------------------------------------')
        # print(f'visual_features_shape:{visual_features.shape}')
        # print(visual_features.shape)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        # text_features_ori = self.encode_textprompt(visual_features)  # 传递视频特征给encode_textprompt
        #
        # text_features = text_features_ori
        # logits_attn = logits1.permute(0, 2, 1)
        # visual_attn = logits_attn @ visual_features
        # visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        # visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[1], visual_attn.shape[2])
        # # text_features = text_features_ori.unsqueeze(0)
        # # print(visual_attn.shape)
        # # print(text_features.shape)
        # text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        # text_features = text_features + visual_attn
        # text_features = text_features + self.mlp1(text_features)
        #
        # visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        # text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features_norm = text_features_norm.permute(0, 2, 1)
        # logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07
        # 生成 prompts
        prompts = self.prompt_learner(visual_features)

        # print(f'prompts shape is {prompts.shape}')
        # print(f'vis_fea shape is {visual_features.shape }')

        # 初始化 logits 列表
        logits = []
        text_features_ori = []
        for pts_i, imf_i in zip(prompts, visual_features):
            imf_i = imf_i.unsqueeze(0)
            # print(f'imf shape is {imf_i.shape}')
            logits_1 = self.classifier(imf_i + self.mlp2(imf_i))
            # print(f'logit1 shape is {logits1.shape}')
            # # 编码文本特征
            text_features = self.clipmodel.encode_text(pts_i, self.prompt_learner.tokenized_prompts)
            text_features_ori.append(text_features)
            # print(f'text_features shape is {text_features.shape}')
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化文本特征
            #
            # # 计算 logits
            # l_i = self.clipmodel.logit_scale.exp() * imf_i @ text_features.t()
            # logits.append(l_i)

            # # 保持原有的处理顺序，进行后续计算
            # text_features_ori = text_features  # 将原有的text_features替换为新生成的text_features
            logits_attn = logits_1.permute(0, 2, 1)
            visual_attn = logits_attn @ imf_i
            visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
            text_features = text_features.unsqueeze(0)
            visual_attn = visual_attn.expand(visual_attn.shape[0], text_features.shape[1], visual_attn.shape[2])
            # text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
            # print(f'visual_attn.shape is {visual_attn.shape}')
            # print(f'text_features.shape is {text_features.shape}')
            text_features = text_features + visual_attn
            text_features = text_features + self.mlp1(text_features)
        #
            visual_features_norm = imf_i / imf_i.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features_norm.permute(0, 2, 1)
            logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07
            # print(f'vis_shape is {visual_features_norm.shape}')
            # print(f'test_shape is {text_features_norm.shape}')
            # print(f'logits shape is {logits2.shape}')
            logits.append(logits2)
        logits = torch.stack(logits)
        logits = logits.squeeze(1)

        # logits = torch.mean(logits, dim=1)
        # print(f'logits shape is {logits.shape}------------------------')

        # 将列表中的tensor合并为一个张量
        text_features_ori = torch.stack(text_features_ori)
        # print(f'text_features_ori now shape is {text_features_ori.shape}')
        # print(f'text_features_ori.shape:{text_features_ori.shape}')

        # # 如果需要增加一个维度
        # text_features_ori = text_features_ori.unsqueeze(0)

        # # 保持原有的处理顺序，进行后续计算
        # text_features_ori = text_features  # 将原有的text_features替换为新生成的text_features
        # logits_attn = logits1.permute(0, 2, 1)
        # visual_attn = logits_attn @ visual_features
        # visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        # visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[1], visual_attn.shape[2])
        # text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        # text_features = text_features + visual_attn
        # text_features = text_features + self.mlp1(text_features)
        #
        # visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        # text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features_norm = text_features_norm.permute(0, 2, 1)
        # logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits



import torch
import torch.nn as nn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


# class PromptLearner(nn.Module):
#     def __init__(self, classnames, clip_model, n_ctx=4, dtype=torch.float32, device='cuda'):
#         super().__init__()
#         n_cls = len(classnames)
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#         nn.init.normal_(ctx_vectors, std=0.02)
#         prompt_prefix = " ".join(["X"] * n_ctx)
#
#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")
#
#         self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
#
#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = []
#
#         # for i, name in enumerate(classnames):
#         #     if name == 'normal':
#         #         prompt = f"This is a normal video {prompt_prefix} of {name}"
#         #     else:
#         #         prompt = f"This is a abnormal video {prompt_prefix} of {name}"
#         #     prompts.append(prompt)
#         for i, name in enumerate(classnames):
#             prompt = f"This is a video of {prompt_prefix} {name}"
#             prompts.append(prompt)
#         # print(prompts)
#
#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).long().to(device)  # Ensure tokenized_prompts is Long type
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(clip_model.dtype)  # Ensure embedding is on the same device
#
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
#
#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts.to(clip_model.dtype)  # Ensure tokenized_prompts is on the same device and dtype
#         self.name_lens = name_lens
#         self.class_token_position = "end"
#
#     def forward(self):
#         ctx = self.ctx
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
#
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#
#         if self.class_token_position == "end":
#             prompts = torch.cat(
#                 [
#                     prefix,  # (n_cls, 1, dim)
#                     ctx,     # (n_cls, n_ctx, dim)
#                     suffix,  # (n_cls, *, dim)
#                 ],
#                 dim=1,
#             )
#         else:
#             raise ValueError("Invalid class token position")
#
#         return prompts.to(self.ctx.device)  # Ensure prompts are on the same device
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=4, dtype=torch.float32, device='cuda'):
        super().__init__()
        self.n_cls = len(classnames)  # 类别数目
        self.ctx_dim = clip_model.ln_final.weight.shape[0]  # 获取上下文向量的维度
        # self.ctx = nn.Parameter(torch.empty(n_ctx, self.ctx_dim, dtype=dtype))  # 共享的可学习上下文向量
        self.ctx = nn.Parameter(torch.empty(n_ctx, self.ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx, std=0.02)  # 初始化上下文向量
        self.device = device  # 设备

        # Meta-Net部分
        vis_dim = clip_model.visual.output_dim  # 获取视觉特征的维度
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),  # 第一层线性变换，将维度缩小到原来的1/16
            ("relu", nn.ReLU(inplace=True)),  # ReLU激活函数
            ("linear2", nn.Linear(vis_dim // 16, self.ctx_dim))  # 第二层线性变换，将维度转换为上下文向量的维度
        ]))

        prompt_prefix = " ".join(["X"] * n_ctx)  # 初始化上下文的文本表示

        prompts = []
        classnames = [name.replace("_", " ") for name in classnames]  # 替换类别名中的下划线
        # self.prompts = [f"This is a video of {prompt_prefix} {name}." for name in classnames]  # 生成prompt模板
        for i, name in enumerate(classnames):
            if name == 'normal':
                prompt = f"This is a normal video of {name} {prompt_prefix}"
            else:
                prompt = f"This is a abnormal video of {name} {prompt_prefix}"
            prompts.append(prompt)

        self.prompts = prompts
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in self.prompts]).long().to(device)  # 将prompt模板进行tokenize

        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_prompts).type(dtype).to(clip_model.dtype)  # 获取token的嵌入表示

        self.register_buffer("token_prefix", embedding[:, :1, :])  # 保存SOS token的嵌入表示
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # 保存CLS和EOS token的嵌入表示

        self.n_ctx = n_ctx  # 上下文向量的长度
        self.class_token_position = "end"  # 类别token的位置

    def forward(self, image_features):

        prefix = self.token_prefix  # 获取保存的SOS token嵌入表示
        suffix = self.token_suffix  # 获取保存的CLS和EOS token嵌入表示
        ctx = self.ctx  # 共享的learnable prompt
        # print('-----------------------------------------------------------------')
        # print(f'ctx_shape:{ctx.shape}')
        image_features = image_features.mean(dim=1)  # (batch_size, ctx_dim)
        # print(f'image_features shape is {image_features.shape}')
        # print('-------------------------------------------------------------------------------------------------------')
        # print(f'The shape of  image_features:{image_features.shape}')
        bias = self.meta_net(image_features)  # 根据视频特征生成的偏移向量
        # print('-----------------------------------------------------------------')
        # print(f'bias_shape:{bias.shape}')
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        # print('-----------------------------------------------------------------')
        # print(f'bias_shape:{bias.shape}')
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # 动态调整的上下文向量
        # print(f'ctx_shifted shape is {ctx_shifted.shape}')

        # 使用instance-conditioned context tokens
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)  # 将调整后的上下文向量扩展到每个类别
            pts_i = torch.cat([prefix, ctx_i, suffix], dim=1)
            # print('------------------------------------------------------------')# 连接prefix、上下文和suffix
            # print(f'The shape of  pts_i:{pts_i.shape}')
            prompts.append(pts_i)
            # print(f'prompt_shape is {pts_i.shape}')
        prompts = torch.stack(prompts)  # 将所有的prompts堆叠起来
        # print(f'prompts_shape is {prompts.shape}')
        # print('-----------------------------------------------------------------')
        # print(f'prompts_shape:{prompts.shape}')

        return prompts.to(self.device)  # 返回prompt并确保其在正确的设备上







