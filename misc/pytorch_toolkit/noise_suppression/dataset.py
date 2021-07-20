"""
 Copyright (c) 2021 Intel Corporation

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

import logging
import os
import time
import random
import multiprocessing

import torch
import numpy as np
import wave

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} dataset'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))


EPS = torch.finfo(torch.float32).tiny

#it is supposed that all data has 16kHz
FREQ = 16000

#set True to speedup (cache) dataset scaning
DUMP_FILE_INFO_FLAG = False

class AudioFile:
    DUMP_VER = "v3"

    def get_dump_str(self):
        return "{} {} {}\n".format(self.freq, self.size, self.file_name)

    def __init__(self, file_name=None, dump_str=None):
        if file_name:
            with wave.open(file_name, "rb") as wav:
                self.freq = wav.getframerate()
                self.size = wav.getnframes()
                self.file_name = file_name
        elif dump_str:
            dump_list = dump_str.strip().split(' ',2)
            self.freq = int(dump_list[0])
            self.size = int(dump_list[1])
            self.file_name = dump_list[2]
        else:
            raise RuntimeError("One of AudioFile args (file_name or dump_str) has to be no None")

        if self.freq != FREQ:
            msg = "freq {}!={} for file {}".format(self.freq, FREQ, file_name if file_name else dump_str)
            printlog(msg)
            raise RuntimeError(msg)

    def read_segment(self, start, stop):
        with wave.open(self.file_name, "rb") as wav:
            if start>0:
                wav.setpos(start)
            data = wav.readframes(stop-start)
            if wav.getsampwidth()==2:
                sample = np.frombuffer(data, dtype=np.int16)
                sample = sample.astype(np.float32)* (1.0 / np.iinfo(np.int16).max)
                if wav.getnchannels()>1:
                    sample = sample.reshape(-1, wav.getnchannels())
                    sample = sample.mean(1)
            else:
                raise RuntimeError("file {} has unsupported sample size {}".format(self.file_name, wav.getsampwidth()))
        return torch.Tensor(sample)

    def read_random_segment(self, size_to_read):
        if self.size >= size_to_read:
            start = random.randint(0, self.size - size_to_read)
            stop = start + size_to_read
            sample = self.read_segment(start, stop)
        else:
            sample = self.read_segment(0, self.size)
            pad = torch.zeros(size_to_read - self.size, dtype=torch.float32)
            sample = torch.cat([pad,sample],-1)
        return sample

    def read_all(self):
        return self.read_segment(0,self.size)

def create_file_info_task(fn):
    try:
        fi = AudioFile(file_name=fn)
    except RuntimeError:
        printlog("fail to read file", fn)
        fi = None
    except wave.Error:
        printlog("fail to read file", fn)
        fi = None
    return fi


def scan_files(dataset_dir):
    t0 = time.time()

    dump_file = os.path.join(dataset_dir, "fileinfo." + AudioFile.DUMP_VER + ".txt")

    if DUMP_FILE_INFO_FLAG and os.path.isfile(dump_file):
        printlog("read dataset file info from dump file", dump_file)
        with open(dump_file, "rt") as inp:
            files = [AudioFile(dump_str=l) for l in inp.readlines()]
    else:
        file_names = []
        printlog("scan dir {} to find audio files".format(dataset_dir))
        for r, _, files in os.walk(dataset_dir):
            for f in files:
                if f.split('.')[-1] in ['wav']:
                    file_names.append(os.path.join(r, f))

        if file_names:
            file_names = sorted(file_names)
            printlog("scan", len(file_names), 'files from', dataset_dir)
            with multiprocessing.Pool() as pool:
                files = pool.map(create_file_info_task, file_names)
                files = [f for f in files if f is not None]
        else:
            raise RuntimeError("can not create list of files for dataset_dir", dataset_dir)

        if DUMP_FILE_INFO_FLAG:
            printlog("write dataset file info to dump file", dump_file)
            with open(dump_file,"wt") as out:
                for fi in files:
                    out.write(fi.get_dump_str())

    total = sum(f.size / f.freq for f in files)
    t1 = time.time()
    printlog("{:.1f} data hours scanned by {:.1f}s from {}".format(total / 60 / 60, t1 - t0, dataset_dir))
    return files


class DNSDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dns_datasets,
                 size_to_read,
                 zero_signal_iter=5,
                 zero_signal_level=-40,
                 non_stationary_noise_iter=5):
        self.non_stationary_noise_iter = non_stationary_noise_iter
        self.zero_signal_iter = zero_signal_iter
        self.zero_signal_level = zero_signal_level
        self.size_to_read = size_to_read

        self.files_clean = scan_files(os.path.join(dns_datasets, "clean"))
        self.files_noise = scan_files(os.path.join(dns_datasets, "noise"))

        def make_idx(files):
            idx = []
            for f_idx, f in enumerate(files):
                idx += [f_idx] * max(1, f.size // self.size_to_read)
            random.shuffle(idx)
            return idx

        self.idx_clean = make_idx(self.files_clean)
        self.idx_noise = make_idx(self.files_noise)

    def __len__(self):
        return len(self.idx_clean)

    def __getitem__(self, sample_i):

        f_clean_idx = self.idx_clean[sample_i]
        f_clean = self.files_clean[f_clean_idx]

        x_clean = f_clean.read_random_segment(self.size_to_read)
        zsi = self.zero_signal_iter
        ms_threshold = 10 ** (self.zero_signal_level / 10)# signal less than -40 db is empty
        while zsi>0:
            #estimate power of signal if it is less than -40dB then try to get another part
            if x_clean.pow(2).mean() < ms_threshold:
                zsi = zsi - 1
                x_clean = f_clean.read_random_segment(self.size_to_read)
            else:
                break

        #sample noise signal
        def get_noise():
            f_noise_idx = random.choice(self.idx_noise)
            x = self.files_noise[f_noise_idx].read_random_segment(self.size_to_read)
            return x

        x_noise = get_noise()

        nsni = self.non_stationary_noise_iter
        # iterate to find nonstantionary noise
        while nsni>0:
            rms_wnd_size = int(0.050 * FREQ) #50ms window
            rms2 = torch.nn.functional.avg_pool1d(
                x_noise.pow(2)[None,None,:],
                rms_wnd_size,
                stride=rms_wnd_size//2)
            db = 10 * torch.log10(rms2+EPS)
            db_std = torch.std(db)
            if db_std.item() < 3:
                nsni = nsni - 1
                x_noise = get_noise()
            else:
                break

        return x_noise, x_clean
