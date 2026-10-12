# Copyright 2025 Cisco Systems, Inc. and its affiliates
# Apache-2.0

import torch
import torch.nn as nn
from .wavlm.feature_extractor import WavLM_feat
from .flow.cfm import CFM
from .vocoder.vocos.vocoder import VocosVocoder


class StuPASE(nn.Module):
    def __init__(
        self, 
        dewavlm_ckpt_path="/work/user_data/xiaobin/Pre-trained/StuPASE/DeWavLM-R.tar",
        cfm_ckpt_path="/work/user_data/xiaobin/Pre-trained/StuPASE/CFM.tar",
        vocoder_ckpt_path="/work/user_data/xiaobin/Pre-trained/StuPASE/Vocoder_Mel-16k.tar",
    ):
        super().__init__()
        self.dewavlm = WavLM_feat(dewavlm_ckpt_path)
        self.cfm = CFM.from_pretrained(cfm_ckpt_path)
        self.vocoder = VocosVocoder.from_pretrained(vocoder_ckpt_path)

    @torch.no_grad()
    def forward(self, x, sr=16000, steps=8, cfg_strength=0.5, sway_sampling_coef=-1.0):
        """
        Args:
            x (torch.Tensor): noisy speech with shape of (B, L) or (B, 1, L)
            sr (int): sampling rate of the input speech
        Return:
            y (torch.Tensor): enhanced speech with shape of (B, L).
        """
        if x.ndim == 3:
            x = x.squeeze(1)  # (B, L)
            
        n_samples = x.shape[-1]
        
        rep = self.dewavlm(x, sr=sr)  # enhanced phonetic representations
        y, _ = self.cfm.sample(
            cond=x, cond_noisy=x, text=rep, no_ref_audio=True, vocoder=self.vocoder,
            steps=steps, cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef
        )
        
        y = y.squeeze(1)  # (B, T)
 
        if y.shape[-1] < n_samples:
            y = nn.functional.pad(y, (0, n_samples - y.shape[-1]), mode='constant', value=0.0)
        else:
            y = y[..., :n_samples]
 
        return y


    
if __name__ == "__main__":

    model = StuPASE()

    from ptflops import get_model_complexity_info
    
    with torch.inference_mode():
        macs, params = get_model_complexity_info(model, (16000,), print_per_layer_stat=False)
    
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(macs, f"{params / 1e6:.2f} M")

