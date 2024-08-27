import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model_test4 import CLIPVAD
from test4_test import test
from utils.dataset import UCFDataset
from utils.tools_test4 import get_prompt_text, get_batch_label
import test4_option
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='logs/test4')

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        # print(f'length is {lengths[i]}')
        # print(f'k is {int(lengths[i] / 16 + 1)}')
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        # print(f'tmp shape is {tmp.shape}')
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    # print(f'logits shape is {logits.shape}')
    # print(f'labels shape is {labels.shape}')
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


# def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
#     model.to(device)
#     gt = np.load(args.gt_path)
#     gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
#     gtlabels = np.load(args.gt_label_path, allow_pickle=True)
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
#     scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
#     ap_best = 0
#     best_mAP = 0
#     epoch = 0
#     best_AUC = 0
#     best_AP = 0
#     best_mAP_ = [0, 0, 0, 0, 0]
#
#     if args.use_checkpoint == True:
#         checkpoint = torch.load(args.checkpoint_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         model.prompt_learner.ctx = checkpoint['prompt_learner_ctx']
#         model.prompt_learner.token_prefix = checkpoint['prompt_learner_token_prefix']
#         model.prompt_learner.token_suffix = checkpoint['prompt_learner_token_suffix']
#         epoch = checkpoint['epoch']
#         ap_best = checkpoint['ap']
#         print("checkpoint info:")
#         print("epoch:", epoch + 1, " ap:", ap_best)
#
#     for e in range(args.max_epoch):
#         initial_ctx = model.prompt_learner.ctx.clone()
#         model.train()
#         loss_total1 = 0
#         loss_total2 = 0
#         normal_iter = iter(normal_loader)
#         anomaly_iter = iter(anomaly_loader)
#         last_test_feature = torch.tensor(1)
#         if not torch.equal(initial_ctx, model.prompt_learner.ctx):
#             print(f"Epoch {e}: ctx has been updated.")
#         else:
#             print('Changed!')
#
#
#
#         for i in range(min(len(normal_loader), len(anomaly_loader))):
#             step = 0
#             normal_features, normal_label, normal_lengths = next(normal_iter)
#             anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)
#
#             visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
#             text_labels = list(normal_label) + list(anomaly_label)
#             feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
#             text_labels = get_batch_label(text_labels, list(label_map.values()), label_map).to(device)
#             # print(label_map[text_labels])
#
#             text_features, logits1, logits2 = model(visual_features, None, list(label_map.values()), feat_lengths)
#             last_test_feature = last_test_feature.to(device)
#             text_features = text_features.to(device)
#             if torch.equal(last_test_feature, text_features):
#                 print('------------------------------------------------------------------------------------------')
#             else:
#                 last_test_feature = text_features
#                 print('good')
#
#             # if i % 8 == 0:
#                 # print(f"Epoch {epoch}: ctx = {model.prompt_learner.ctx.shape}")
#
#             # loss1
#             loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
#             loss_total1 += loss1.item()
#
#             # loss2
#             loss2 = CLASM(logits2, text_labels, feat_lengths, device)
#             loss_total2 += loss2.item()
#             # #loss3
#             # loss3 = torch.zeros(1).to(device)
#             # text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
#             # for j in range(1, text_features.shape[0]):
#             #     text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
#             #     loss3 += torch.abs(text_feature_normal @ text_feature_abr)
#             # loss3 = loss3 / 13 * 1e-1
#             #
#             # loss = loss1 + loss2 + loss3
#             # loss = loss1
#             loss = loss1 + loss2
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             step += i * normal_loader.batch_size * 2
#             if step % 1280 == 0 and step != 0:
#                 # print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
#                 print('epoch: ', e + 1, '| step: ', step, '| loss1: ', loss_total1 / (i + 1), '| loss2: ',
#                       loss_total2 / (i + 1))
#                 # print('epoch: ', e + 1, '| step: ', step, '| loss1: ', loss_total1 / (i + 1))
#                 map_ = []
#                 AUC, AP, mAP, _best_mAP, map_ = test(model, testloader, args.visual_length, list(label_map.values()), gt, gtsegments, gtlabels, device, best_mAP)
#                 writer.add_scalar('AUC', AUC, e * len(normal_loader) + i)
#                 writer.add_scalar('AP', AP, e * len(normal_loader) + i)
#                 writer.add_scalar('mAP', mAP, e * len(normal_loader) + i)
#                 writer.add_scalar('mAP@0.1', map_[0], e * len(normal_loader) + i)
#                 writer.add_scalar('mAP@0.2', map_[1], e * len(normal_loader) + i)
#                 writer.add_scalar('mAP@0.3', map_[2], e * len(normal_loader) + i)
#                 writer.add_scalar('mAP@0.4', map_[3], e * len(normal_loader) + i)
#                 writer.add_scalar('mAP@0.5', map_[4], e * len(normal_loader) + i)
#
#                 if _best_mAP > best_mAP:
#                     best_mAP = _best_mAP
#                     best_mAP_[0] = map_[0]
#                     best_mAP_[1] = map_[1]
#                     best_mAP_[2] = map_[2]
#                     best_mAP_[3] = map_[3]
#                     best_mAP_[4] = map_[4]
#                 if AUC > best_AUC:
#                     best_AUC = AUC
#                 if AP > best_AP:
#                     best_AP = AP
#
#                 AP = AUC
#
#                 if AP > ap_best:
#                     print(f'best AUC is {AP}')
#                     ap_best = AP
#                     checkpoint = {
#                         'epoch': e,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'ap': ap_best,
#                         'prompt_learner_ctx': model.prompt_learner.ctx,
#                         'prompt_learner_token_prefix': model.prompt_learner.token_prefix,
#                         'prompt_learner_token_suffix': model.prompt_learner.token_suffix
#                     }
#                     torch.save(checkpoint, args.checkpoint_path)
#
#
#
#         scheduler.step()
#         # if AP > ap_best:
#         print(f'Best AUC is {best_AUC}')
#         print(f'Best mAP is {best_mAP}')
#         print(f'mAP_list: {best_mAP_}')
#
#         torch.save(model.state_dict(), '../model_test3/model_cur.pth')
#         checkpoint = torch.load(args.checkpoint_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#
#     checkpoint = torch.load(args.checkpoint_path)
#     torch.save(checkpoint['model_state_dict'], args.model_path)
#     writer.close()
def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    ap_best = 0
    best_mAP = 0
    epoch = 0
    best_AUC = 0
    best_AP = 0
    best_mAP_ = [0, 0, 0, 0, 0]

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.prompt_learner.ctx = checkpoint['prompt_learner_ctx']
        model.prompt_learner.token_prefix = checkpoint['prompt_learner_token_prefix']
        model.prompt_learner.token_suffix = checkpoint['prompt_learner_token_suffix']
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        model.prompt_learner.meta_net.linear1.weight = checkpoint['model.prompt_learner.meta_net.linear1.weight']
        model.prompt_learner.meta_net.linear1.bias = checkpoint['model.prompt_learner.meta_net.linear1.bias']
        model.prompt_learner.meta_net.linear2.weight = checkpoint['model.prompt_learner.meta_net.linear2.weight']
        model.prompt_learner.meta_net.linear2.bias = checkpoint['model.prompt_learner.meta_net.linear2.bias']
        print("checkpoint info:")
        print("epoch:", epoch + 1, " ap:", ap_best)

    for e in range(args.max_epoch):
        initial_ctx = model.prompt_learner.ctx.clone()
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        last_test_feature = torch.tensor(1)

        for i in range(min(len(normal_loader), len(anomaly_loader))):
            # print(f'i:{i}')
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, list(label_map.values()), label_map).to(device)

            text_features, logits1, logits2 = model(visual_features, None, list(label_map.values()), feat_lengths)
            last_test_feature = last_test_feature.to(device)
            text_features = text_features.to(device)
            # if torch.equal(last_test_feature, text_features):
            #     print('------------------------------------------------------------------------------------------')
            # else:
            #     last_test_feature = text_features
            #     print('good')
            #
            # if i % 8 == 0:
            #     print(f"Epoch {epoch}: ctx = {model.prompt_learner.ctx.shape}")

            # loss1
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()

            # loss2
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            # loss3 = torch.zeros(1).to(device)
            #
            # text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            # for j in range(1, text_features.shape[0]):
            #     text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
            #     # print(text_feature_normal.shape)
            #     # print(text_feature_abr.shape)
            #     loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            # loss3 = torch.zeros(1).to(device)
            # text_features_normalized = text_features / text_features.norm(dim=-1, keepdim=True)
            #
            # for i in range(text_features.shape[0]):
            #     text_feature_normal = text_features_normalized[i]  # (14, 512)
            #     for j in range(text_features.shape[0]):
            #         if i != j:
            #             text_feature_abr = text_features_normalized[j]  # (14, 512)
            #             # (14, 512) @ (512, 14) -> (14, 14) -> mean() -> scalar
            #             loss3 += torch.abs((text_feature_normal @ text_feature_abr.transpose(-1, -2)).mean())
            #
            # loss3 = loss3 / (text_features.shape[0] * (text_features.shape[0] - 1)) * 1e-1
            loss3 = torch.zeros(1).to(device)

            # 归一化 text_features
            text_features_normalized = text_features / text_features.norm(dim=-1, keepdim=True)  # (batch_size, 14, 512)

            # 遍历 batch 中的每个样本
            for batch_idx in range(text_features.shape[0]):
                for q in range(text_features.shape[1]):
                    text_feature_normal = text_features_normalized[batch_idx, q]  # (512)
                    for j in range(text_features.shape[1]):
                        if q != j:
                            text_feature_abr = text_features_normalized[batch_idx, j]  # (512)
                            # 计算 (512) @ (512) -> scalar
                            loss3 += torch.abs((text_feature_normal @ text_feature_abr).mean())

            # 计算平均 loss3
            loss3 = loss3 / (text_features.shape[0] * text_features.shape[1] * (text_features.shape[1] - 1)) * 1e-1

            # loss3 = loss3 / 13 * 1e-1
            #
            # loss = loss1 + loss2 + loss3
            # loss = loss1
            loss = loss1 + loss2 + loss3
            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(i)
            step += i * normal_loader.batch_size * 2
            # print(f'step is {step}')
            # print(f'normal_loader.batch_size is {normal_loader.batch_size}')
            # print(i * normal_loader.batch_size * 2)
            if step % 1280 == 0 and step != 0:

                # print('epoch: ', e + 1, '| step: ', step, '| loss1: ', loss_total1 / (i + 1), '| loss2: ',
                #       loss_total2 / (i + 1))
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                map_ = []
                AUC, AP, mAP, _best_mAP, map_ = test(model, testloader, args.visual_length, list(label_map.values()),
                                                     gt, gtsegments, gtlabels, device, best_mAP)
                # AUC, AP, mAP, _best_mAP, map_ = test(model, testloader, args.visual_length, prompt_text,
                #                                      gt, gtsegments, gtlabels, device, best_mAP)
                writer.add_scalar('AUC', AUC, e * len(normal_loader) + i)
                writer.add_scalar('AP', AP, e * len(normal_loader) + i)
                writer.add_scalar('mAP', mAP, e * len(normal_loader) + i)
                writer.add_scalar('mAP@0.1', map_[0], e * len(normal_loader) + i)
                writer.add_scalar('mAP@0.2', map_[1], e * len(normal_loader) + i)
                writer.add_scalar('mAP@0.3', map_[2], e * len(normal_loader) + i)
                writer.add_scalar('mAP@0.4', map_[3], e * len(normal_loader) + i)
                writer.add_scalar('mAP@0.5', map_[4], e * len(normal_loader) + i)

                if _best_mAP > best_mAP:
                    best_mAP = _best_mAP
                    best_mAP_[0] = map_[0]
                    best_mAP_[1] = map_[1]
                    best_mAP_[2] = map_[2]
                    best_mAP_[3] = map_[3]
                    best_mAP_[4] = map_[4]
                if AUC > best_AUC:
                    best_AUC = AUC
                if AP > best_AP:
                    best_AP = AP

                AP = AUC

                # print(f'AP is {AP}')
                # print(f'ap_best is {ap_best}')
                if AP > ap_best or AP == ap_best:

                    print(f'best AUC is {AP} ')
                    ap_best = AP
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best,
                        # 'prompt_learner_ctx': model.prompt_learner.ctx,
                        # 'prompt_learner_token_prefix': model.prompt_learner.token_prefix,
                        # 'prompt_learner_token_suffix': model.prompt_learner.token_suffix,
                        # 'prompt_learner.meta_net.linear1.weight':model.prompt_learner.meta_net.linear1.weight,
                        # 'prompt_learner.meta_net.linear1.bias':model.prompt_learner.meta_net.linear1.bias,
                        # 'prompt_learner.meta_net.linear2.weight':model.prompt_learner.meta_net.linear2.weight,
                        # 'prompt_learner.meta_net.linear2.bias':model.prompt_learner.meta_net.linear2.bias
                    }
                    torch.save(checkpoint, args.checkpoint_path)

        scheduler.step()
        print(f'Best AUC is {best_AUC}')
        print(f'Best mAP is {best_mAP}')
        print(f'mAP_list: {best_mAP_}')



    #     torch.save(model.state_dict(), '../model_test4/model_cur.pth')
    #     checkpoint = torch.load(args.checkpoint_path)
    #     print(checkpoint.keys())
    #
    #     # state_dict = checkpoint['model_state_dict']
    #     # print(state_dict.keys())
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     # model.prompt_learner.ctx = checkpoint['prompt_learner_ctx']
    #     # model.prompt_learner.token_prefix = checkpoint['prompt_learner_token_prefix']
    #     # model.prompt_learner.token_suffix = checkpoint['prompt_learner_token_suffix']
    #     # model.prompt_learner.meta_net.linear1.weight = checkpoint['prompt_learner.meta_net.linear1.weight']
    #     # model.prompt_learner.meta_net.linear1.bias = checkpoint['prompt_learner.meta_net.linear1.bias']
    #     # model.prompt_learner.meta_net.linear2.weight = checkpoint['prompt_learner.meta_net.linear2.weight']
    #     # model.prompt_learner.meta_net.linear2.bias = checkpoint['prompt_learner.meta_net.linear2.bias']
    #
    # checkpoint = torch.load(args.checkpoint_path)
    #
    # torch.save(checkpoint['model_state_dict'], args.model_path)
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)
    writer.close()


#
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = test4_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault',
                      'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting',
                      'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting',
                      'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head,
                    args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, label_map, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)

