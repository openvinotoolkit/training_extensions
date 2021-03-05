"""
 Copyright (c) 2020 Intel Corporation
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
import argparse
import os
# torch
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
# model
from transformers import MarianMTModel, MarianTokenizer
# dataset
from core.dataset.lmdb_container import LMDBContainer, Page
from core.dataset.text_container import TextContainer
from torch.utils.data import DataLoader
# lmdb
import lmdb
# utils
from core.utils import all_gather
from tqdm.autonotebook import tqdm


class Tokenizer:
    def __init__(self, langs):
        self.fn = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{langs}')

    def __call__(self, batch_in):
        batch_out = {"text": [], "key": []}
        for sample in batch_in:
            batch_out["key"].append(sample["key"])
            batch_out["text"].append(sample["text"])
        tokens = self.fn.prepare_translation_batch(batch_out["text"])
        for k, v in tokens.items():
            batch_out[k] = v
        return batch_out

    def decode_batch(self, batch):
        print(batch)
        s = self.fn.decode(batch[0].tolist())
        return s


def main(gpu, args):
    ############################################################
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
    	backend='nccl',
   	init_method='env://',
    	world_size=args.world_size,
    	rank=rank
    )
    ############################################################
    # Open target LMDB (rank = 0)
    ############################################################
    if rank == 0:
        print(f"[{rank}] load lmdb...")
        env = lmdb.open(args.output_lmdb, map_size=args.lmdb_map_size)
        page = Page(env, 1000)

    torch.manual_seed(42)
    torch.cuda.set_device(gpu)
    ############################################################
    # Model
    ############################################################
    print(f"[{rank}] load model...")
    model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{args.langs}').cuda(gpu)
    ############################################################
    # Dataset
    ############################################################
    print(f"[{rank}] load dataset...")
    if args.input_format == "lmdb":
        dataset = LMDBContainer(args.input_corpus)
    else:
        dataset = TextContainer(args.input_corpus)
    tokenizer = Tokenizer(args.langs)
    sampler = torch.utils.data.distributed.DistributedSampler(
    	dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler,
        collate_fn=tokenizer
    )
    ############################################################
    # Translation loop
    ############################################################
    print(f"[{rank}] run loop...")
    for batch in tqdm(loader):
        sample = {}
        for k in ['input_ids', 'attention_mask']:
            sample[k] = batch[k].cuda(gpu)
        with torch.no_grad():
            preds = model.generate(**sample)
        text = tokenizer.fn.batch_decode(preds, skip_special_tokens=True)
        text = all_gather(text)
        key = all_gather(batch["key"])
        if rank == 0:
            for i in range(len(text)):
                for j in range(len(text[i])):
                    page.add(text[i][j], key[i][j])
    if rank == 0:
        page.close()
    ############################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('-l', '--langs', default="ru-en", type=str, help="languages pair from available for MarianMTModel")
    parser.add_argument('--batch-size', type=int, default=32, help="batch size")
    parser.add_argument('--num-workers', type=int, default=0, help="num workers")
    # LMDB
    parser.add_argument('--input-corpus', type=str, help="path to input corpus")
    parser.add_argument('--input-format', type=str, default="lmdb", help="format of input corpus [txt|lmdb]")
    parser.add_argument('--output-lmdb', type=str, help="path to output lmdb file")
    parser.add_argument('--lmdb-map-size', type=int, default=10**12, help="lmdb map size")
    parser.add_argument('--lmdb-page-size', type=int, default=1000, help="lmdb page size")
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    #########################################################
    mp.spawn(main, nprocs=args.gpus, args=(args,))
