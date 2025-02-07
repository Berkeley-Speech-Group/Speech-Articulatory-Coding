import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from pathlib import Path
import random
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import soundfile as sf
import librosa

def restore_features(segments, segfts,end_buffer=10):
    segfts = np.concatenate([np.zeros((1,segfts.shape[-1])),
                                 segfts],0)
    ft_len = np.max(segments)+end_buffer

    seg_idxs = np.zeros(ft_len)
    progress = np.zeros(ft_len)
    for seg_i, (s,e) in enumerate(segments):
        e = min(ft_len, e)
        if e<=s:
            continue
        seg_idxs[s:e]=seg_i+1
        progress[s:e] = np.linspace(0,1,e-s+1)[1:] 

    features = segfts[seg_idxs.astype(int)]
    return features, progress

        
        
class SpeechDataset(Dataset):
    
    def __init__(self, wav_files, seg_files, ft_files, spkemb_files, sample_len=1, target_sr=16000, ft_sr=50, 
                normalize=True, normalize_rescale=0.05):
        super().__init__()
        self.wav_files = wav_files
        self.seg_files = seg_files
        self.ft_files = ft_files
        self.spkemb_files = spkemb_files
        self.sample_len = sample_len
        self.target_sr = target_sr
        self.ft_sr = ft_sr
        self.normalize = normalize
        self.normalize_rescale = normalize_rescale
    
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self,i):
        wav_file, seg_file, ft_file, spkemb_file = self.wav_files[i], self.seg_files[i], self.ft_files[i], self.spkemb_files[i]
        wav_file = self.wav_files[i]
        wav,sr = sf.read(wav_file)
        if sr != self.target_sr:
            wav = librosa.resample(wav, orig_sr=sr,
                                   target_sr=self.target_sr)
            sr = self.target_sr
        if self.normalize:
            wav = (wav-wav.mean())/wav.std()*self.normalize_rescale #0.05
        #wav = np.concatenate([np.zeros(160), wav, np.zeros(160)])
        
        segfts = np.load(ft_file)
        segfts = np.concatenate([np.zeros((1,segfts.shape[-1])),
                                 segfts],0)
        segments = np.load(seg_file)
        spkemb = np.load(spkemb_file)

        ft_len = int(np.ceil(len(wav)/sr*self.ft_sr))

        seg_idxs = np.zeros(ft_len)
        progress = np.zeros(ft_len)
        for seg_i, (s,e) in enumerate(segments):
            e = min(ft_len, e)
            if e<=s:
                continue
            seg_idxs[s:e]=seg_i+1
            progress[s:e] = np.linspace(0,1,e-s+1)[1:] 
        
        ft_sample_len = int(self.sample_len/self.target_sr*self.ft_sr)
        if len(seg_idxs)>ft_sample_len:
            p = np.random.randint(len(seg_idxs)-ft_sample_len)
            seg_idxs = seg_idxs[p:p+ft_sample_len]
            progress = progress[p:p+ft_sample_len]
            wav_p = int(p*sr/self.ft_sr)
            wav = wav[wav_p:wav_p+self.sample_len] #TODO add condition for wavlen matching
            
        features = torch.from_numpy(segfts[seg_idxs.astype(int)]).float()
        progress = torch.from_numpy(progress).float()
        wav = torch.from_numpy(wav).float()
        spkemb = torch.from_numpy(spkemb).float()
        return {'wav':wav, 'features':features, 'progress':progress, 'spkemb':spkemb}
    
    @staticmethod
    def collate(batch):
        data = {}
        data['wav'] = nn.utils.rnn.pad_sequence([d['wav'] for d in batch], batch_first=True, padding_value=0.0)
        data['features'] = nn.utils.rnn.pad_sequence([d['features'] for d in batch], batch_first=True, padding_value=0.0)
        data['progress'] = nn.utils.rnn.pad_sequence([d['progress'] for d in batch], batch_first=True, padding_value=0.0)
        data['spkemb'] = torch.stack([d['spkemb'] for d in batch])
        return data

class SpeechDataModule(LightningDataModule):
    def __init__(self,
                 train_datasets,
                 val_datasets,
                 sample_len=5120,
                 batch_size=64,
                 val_batch_size=None,
                 num_workers=4,
                 drop_last=True,
                 pin_memory=True,
                 normalize=True,  # Whether to z-score audio wave
                 normalize_rescale=0.05, # To reduce the value range.
                 ):
        super().__init__()
        
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.sample_len = sample_len
        
        self.batch_size=batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        
        self.normalize = normalize
        self.normalize_rescale = normalize_rescale
         
        
    def _load_tags(self, split):

        wav_files = []
        art_files = []
        spk_files = []
        datasets = {"train":self.train_datasets, "val":self.val_datasets}[split]
        for dataset_name, dataset in datasets.items():
            print(f"[{split}] Checking {dataset_name}...") 
            wav_dir = Path(dataset['wav_dir'])
            art_dir = Path(dataset['art_dir'])
            spk_dir = Path(dataset['spk_dir'])
            file = dataset['file']
            with open(file, "r") as f:
                file_names = [t.split("|")[0] for t in f.readlines()]
            cnt = 0
            for file_name in file_names:
                wav_file = wav_dir/file_name
                file_tag = file_name.split('.')[0]
                art_file = art_dir/f"{file_tag}.npy"
                spk_file = spk_dir/f"{file_tag}.npy"
                spkemb_file = spkemb_dir/f"{file_tag}.npy"

                if wav_file.exists() and art_file.exists() and spk_file.exists():
                    wav_files.append(wav_file)
                    art_files.append(art_file)
                    spk_files.append(spk_file)
                    cnt+=1
            print(f"[{split}][{dataset_name}] {cnt}/{len(file_names)} files are found.")
        print(f"[{split}] Total {len(wav_files)}")
         
        return wav_files, art_files, spk_files
    
    def train_dataloader(self, batch_size=None):
        file_lists = self._load_tags("train")
        dataset = SpeechDataset(*file_lists, sample_len=self.sample_len, target_sr=16000,
                               normalize=self.normalize, normalize_rescale=self.normalize_rescale)
        batch_size = self.batch_size if batch_size is None else batch_size
        loader = DataLoader(
            dataset,
            batch_size=batch_size ,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechDataset.collate
        )
        return loader
    
    def val_dataloader(self):
        file_lists = self._load_tags("val")
        dataset = SpeechDataset(*file_lists, sample_len=self.sample_len, target_sr=16000,
                               normalize=self.normalize, normalize_rescale=self.normalize_rescale)
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=SpeechDataset.collate
        )
        return loader
    