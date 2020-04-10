"""
 MIT License

 Copyright (c) 2018 Kaiyang Zhou

 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import numpy as np
import time
from os import path as osp
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from torchreid import metrics
from torchreid.engine.image.softmax import ImageSoftmaxEngine
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers, visualize_ranked_results, re_ranking

from .losses.am_softmax import AMSoftmaxLoss
from .losses.cross_entropy_loss import CrossEntropyLoss
from .losses.regularizers import get_regularizer
from .losses.metric import LocalPushLoss


class ImageAMSoftmaxEngine(ImageSoftmaxEngine):
    r"""Softmax-loss engine for image-reid.
    """

    def __init__(self, datamanager, model, optimizer, reg_cfg, metric_cfg, batch_transform_cfg, scheduler=None, use_gpu=False,
                 softmax_type='stock', label_smooth=True, conf_penalty=False,
                 pr_product=False, m=0.35, s=10, writer=None, openvino_model=None, flip_eval=False):
        super(ImageAMSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.regularizer = get_regularizer(reg_cfg)
        self.openvino_model = openvino_model
        self.writer = writer
        self.flip_eval = flip_eval

        if softmax_type == 'stock':
            self.criterion = CrossEntropyLoss(
                num_classes=self.datamanager.num_train_pids,
                use_gpu=self.use_gpu,
                label_smooth=label_smooth,
                conf_penalty=conf_penalty
            )
        elif softmax_type == 'am':
            self.criterion = AMSoftmaxLoss(
                num_classes=self.datamanager.num_train_pids,
                use_gpu=self.use_gpu,
                conf_penalty=conf_penalty,
                m=m, s=s,
                pr_product=pr_product
            )

        self.batch_transform_cfg = batch_transform_cfg
        self.lambd_distr = torch.distributions.beta.Beta(self.batch_transform_cfg.alpha,
                                                         self.batch_transform_cfg.alpha)

        if metric_cfg.enable:
            self.metric_loss = LocalPushLoss(weight=metric_cfg.local_push_weight)
        else:
            self.metric_loss = None

    def run(self, save_dir='log', max_epoch=0, start_epoch=0, fixbase_epoch=0, open_layers=None,
            start_eval=0, eval_freq=-1, test_only=False, print_freq=10,
            dist_metric='euclidean', normalize_feature=False, visrank=False, visrank_topk=10,
            use_metric_cuhk03=False, ranks=(1, 5, 10, 20), rerank=False, visactmap=False):
        r"""A unified pipeline for training and evaluating a model.
        """
        trainloader, testloader = self.datamanager.return_dataloaders()

        if visrank and not test_only:
            raise ValueError('visrank=True is valid only if test_only=True')

        if test_only:
            self.test(
                0,
                testloader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            return

        if visactmap:
            self.visactmap(testloader, save_dir, self.datamanager.width, self.datamanager.height, print_freq)
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        print('=> Start training')

        # Save zeroth checkpoint
        self._save_checkpoint(-1, 0.0, save_dir)

        for epoch in range(start_epoch, max_epoch):
            self.train(epoch, max_epoch, trainloader, fixbase_epoch, open_layers, print_freq)

            if (epoch + 1) >= start_eval and eval_freq > 0 and (epoch + 1) % eval_freq == 0:
                rank1 = self.test(
                    epoch,
                    testloader,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks
                )
                self._save_checkpoint(epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses = AverageMeter()
        reg_ow_loss = AverageMeter()
        metric_loss = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        start_time = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - start_time)

            imgs, pids = self._parse_data_for_train(data)
            imgs, pids = self._apply_batch_transform(imgs, pids)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            self.optimizer.zero_grad()
            if self.metric_loss is not None:
                embeddings, outputs = self.model(imgs, get_embeddings=True)
            else:
                outputs = self.model(imgs)

            loss = self._compute_loss(self.criterion, outputs, pids)

            if (epoch + 1) > fixbase_epoch:
                reg_loss = self.regularizer(self.model)
                reg_ow_loss.update(reg_loss.item(), pids.size(0))
                loss += reg_loss

            if self.metric_loss is not None:
                metric_val = self.metric_loss(F.normalize(embeddings, dim=1),
                                              outputs, pids)
                loss += metric_val
                metric_loss.update(metric_val.item(), pids.size(0))

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())
            batch_time.update(time.time() - start_time)

            if print_freq > 0 and (batch_idx + 1) % print_freq == 0:
                eta_seconds = batch_time.avg * (num_batches-(batch_idx + 1) + (max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'AUX Losses {aux_losses.val:.4f} ({aux_losses.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.
                      format(
                          epoch + 1, max_epoch, batch_idx + 1, num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          aux_losses=metric_loss,
                          loss=losses,
                          acc=accs,
                          lr=self.optimizer.param_groups[0]['lr'],
                          eta=eta_str,
                      )
                )

                if self.writer is not None:
                    n_iter = epoch * num_batches + batch_idx
                    self.writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                    self.writer.add_scalar('Train/Data', data_time.avg, n_iter)
                    info = self.criterion.get_last_info()
                    for k in info:
                        self.writer.add_scalar('AUX info/' + k, info[k], n_iter)
                    self.writer.add_scalar('Loss/train', losses.avg, n_iter)
                    if (epoch + 1) > fixbase_epoch:
                        self.writer.add_scalar('Loss/reg_ow', reg_ow_loss.avg, n_iter)
                    self.writer.add_scalar('Accuracy/train', accs.avg, n_iter)
                    self.writer.add_scalar('Learning rate', self.optimizer.param_groups[0]['lr'], n_iter)
                    if self.metric_loss is not None:
                        self.writer.add_scalar('Loss/local_push_loss',
                                               metric_val.item(), n_iter)
            start_time = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

    @torch.no_grad()
    def _evaluate(self, epoch, dataset_name='', queryloader=None, galleryloader=None,
                  dist_metric='euclidean', normalize_feature=False, visrank=False,
                  visrank_topk=10, save_dir='', use_metric_cuhk03=False, ranks=(1, 5, 10, 20),
                  rerank=False, iteration=0):
        batch_time = AverageMeter()

        print('Extracting features from query set...')
        qf, q_pids, q_camids = [], [], []  # query features, query person IDs and query camera IDs
        for batch_idx, data in tqdm(enumerate(queryloader), 'Processing query...'):
            imgs, pids, camids = self._parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = self._extract_features(imgs, data[3])
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set...')
        gf, g_pids, g_camids = [], [], []  # gallery features, gallery person IDs and gallery camera IDs
        for batch_idx, data in tqdm(enumerate(galleryloader), 'Processing gallery...'):
            imgs, pids, camids = self._parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = self._extract_features(imgs, data[3])
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalizing features with L2 norm...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print('Computing distance matrix with metric={}...'.format(dist_metric))
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )
        if self.writer is not None:
            self.writer.add_scalar('Val/{}/mAP'.format(dataset_name), mAP, epoch + 1)
            for r in ranks:
                self.writer.add_scalar('Val/{}/Rank-{}'.format(dataset_name, r), cmc[r - 1], epoch + 1)

        print('** Results **')
        print('mAP: {:.2%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.2%}'.format(r, cmc[r-1]))

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk
            )

        return cmc[0]

    @torch.no_grad()
    def _extract_features(self, input, img_path=None):
        if self.openvino_model is None:
            self.model.eval()
            out_pytorch = self.model(input)
            if self.flip_eval:
                input_flipped = torch.flip(input, (3,))
                out_pytorch += self.model(input_flipped)
                out_pytorch /= 2.
            return out_pytorch
        else:
            out_openvino = self.openvino_model.forward(img_path)
            out_openvino = np.concatenate(out_openvino)
            out_openvino = torch.tensor(out_openvino).to(input.device)
            return out_openvino

    def _apply_batch_transform(self, imgs, pids):
        if self.batch_transform_cfg.enable:
            permuted_idx = torch.randperm(imgs.shape[0])
            lambd = self.batch_transform_cfg.anchor_bias \
                    + (1 - self.batch_transform_cfg.anchor_bias) \
                    * self.lambd_distr.sample((imgs.shape[0],))
            lambd = lambd.view(-1, 1, 1, 1)
            imgs = lambd * imgs + (1 - lambd) * imgs[permuted_idx]
        return imgs, pids
