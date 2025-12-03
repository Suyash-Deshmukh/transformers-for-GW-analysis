import torch, torch.nn as nn # type: ignore
import torchaudio.transforms as T  # Torch-native resampling # type: ignore
from peft import LoraConfig, get_peft_model # type: ignore
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor # type: ignore
import fnmatch

###########################################################################
# Whisper
###########################################################################

class WhisperBinaryClassifier(nn.Module):
    def __init__(self, model, num_classes=1):
        super().__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.d_model * 2, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, h1_input, l1_input):
        output_h1 = self.model(h1_input).last_hidden_state.mean(dim=1)
        output_l1 = self.model(l1_input).last_hidden_state.mean(dim=1)
        outputs = torch.cat((output_h1, output_l1), dim=1)
        logits = self.classifier(outputs)
        return logits

class WhisperModule(nn.Module):
    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-tiny",
        model_name: str = "whisper",
        input_sample_rate: int = 2048,
        target_sample_rate: int = 16000,
        device: str = "cuda"
    ):
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.whisper_model_name = whisper_model_name
        self.model_name = model_name
        self.device = device

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.whisper_model_name)
        self.whisper_model = AutoModel.from_pretrained(self.whisper_model_name)
        self.whisper_model = self.whisper_model.encoder
        self.resampler = T.Resample(orig_freq=self.input_sample_rate, new_freq=self.target_sample_rate)
        
        lin_names = [n for n, m in self.whisper_model.named_modules() if isinstance(m, nn.Linear)]
        attn_lin  = [n for n in lin_names if ("attn" in n or "attention" in n)]
        cands = ["q_proj","k_proj","v_proj","out_proj","qkv","proj","query","key","value","out"]
        target_modules = [c for c in cands if any(c in n for n in attn_lin)] or \
                        sorted({n.split(".")[-1] for n in attn_lin}) or ["query","key","value","out"]
        print("LoRA target_modules =", target_modules)

        lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=target_modules)
        self.whisper_model = get_peft_model(self.whisper_model, lora_config).to(device)
        self.whisper_model.print_trainable_parameters()

        self.classifier = WhisperBinaryClassifier(self.whisper_model).to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract individual detector data
        x_h1 = x[:, 0, :]  # H1 detector (batch, time)
        x_l1 = x[:, 1, :]  # L1 detector (batch, time)

        # Resample to target sample rate
        x_h1 = self.resampler(x_h1)
        x_l1 = self.resampler(x_l1)

        # Use feature extractor
        x_h1_features = self.feature_extractor(x_h1.cpu().numpy(),
                                                sampling_rate=self.target_sample_rate,
                                                return_tensors="pt").input_features.squeeze().to(self.device)
        x_l1_features = self.feature_extractor(x_l1.cpu().numpy(),
                                                sampling_rate=self.target_sample_rate,
                                                return_tensors="pt").input_features.squeeze().to(self.device)

        # Pass through the classifier head
        logits = self.classifier(x_h1_features, x_l1_features)
        return logits
    
###########################################################################
# Wav2Vec2
###########################################################################

class Wav2Vec2BinaryClassifier(nn.Module):
    def __init__(self, model, num_classes=1):
        super().__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size * 2, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, h1_input, l1_input):
        output_h1 = self.model(h1_input).last_hidden_state.mean(dim=1)
        output_l1 = self.model(l1_input).last_hidden_state.mean(dim=1)
        outputs = torch.cat((output_h1, output_l1), dim=1)
        logits = self.classifier(outputs)
        return logits

class Wav2Vec2Module(torch.nn.Module):
    def __init__(
        self,
        wav2vec2_model_name: str = "facebook/wav2vec2-base-960h",
        model_name: str = "wav2vec2",
        input_sample_rate: int = 2048,
        target_sample_rate: int = 16000,
        device: str = "cuda"
    ):
        super(Wav2Vec2Module, self).__init__()
        self.wav2vec2_model_name = wav2vec2_model_name
        self.model_name = model_name
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.device = device

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.wav2vec2_model_name) #"facebook/wav2vec2-large-960h"
        # self.processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_model_name)
        self.wav2vec2_model = AutoModel.from_pretrained(self.wav2vec2_model_name)
        self.resampler = T.Resample(orig_freq=self.input_sample_rate, new_freq=self.target_sample_rate)
        
        # lin_names = [n for n, m in self.wav2vec2_model.named_modules() if isinstance(m, nn.Linear)]
        # attn_lin  = [n for n in lin_names if ("attn" in n or "attention" in n)]
        # cands = ["q_proj","k_proj","v_proj","out_proj","qkv","proj","query","key","value","out"]
        # target_modules = [c for c in cands if any(c in n for n in attn_lin)] or \
        #                 sorted({n.split(".")[-1] for n in attn_lin}) or ["query","key","value","out"]
        # print("LoRA target_modules =", target_modules)

        # lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=target_modules)
        # self.wav2vec2_model = get_peft_model(self.wav2vec2_model, lora_config).to(device)
        # self.wav2vec2_model.print_trainable_parameters()
        
        attn_modules = []
        for name, _ in self.wav2vec2_model.named_modules():
            if any(x in name for x in ["attn", "attention"]):
                if isinstance(_, nn.Linear):
                    attn_modules.append(name.split(".")[-1])

        # lora_config = LoraConfig(use_dora=False, r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=list(set(attn_modules)))
        # self.wav2vec2_model = get_peft_model(self.wav2vec2_model, lora_config)

        # for n, p in self.wav2vec2_model.named_parameters():
        #     p.requires_grad = ("lora" in n)

        # Apply DoRA via LoRA to the attention layers
        # module_names = [name for name, _ in self.wav2vec2_model.named_modules()]
        # patterns = [
        #     "encoder.layers.*.attention.q_proj",
        #     "encoder.layers.*.attention.k_proj",
        #     "encoder.layers.*.attention.v_proj",
        #     "encoder.layers.*.attention.out_proj",
        # ]
        # matched_modules = []
        # for pattern in patterns:
        #     matched_modules.extend(fnmatch.filter(module_names, pattern))
        lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=32, target_modules=list(set(attn_modules)))#matched_modules)
        
        self.wav2vec2_model = get_peft_model(self.wav2vec2_model, lora_config)
        
        # Freeze base parameters; only train LoRA parameters.
        for name, param in self.wav2vec2_model.named_parameters():
            param.requires_grad = "lora" in name

        self.wav2vec2_model.print_trainable_parameters()

        self.classifier = Wav2Vec2BinaryClassifier(self.wav2vec2_model).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract individual detector data
        if x.dim() == 3 and x.shape[1] == 2:
            x_h1 = x[:, 0, :]  # Channel 1
            x_l1 = x[:, 1, :]  # Channel 2
        else:
            raise ValueError("Input tensor must have shape [batch, 2, time].")
        
        # Resample to target sample rate
        x_h1 = self.resampler(x_h1)
        x_l1 = self.resampler(x_l1)

        # Use feature extractor
        # x_h1_features = self.feature_extractor(x_h1.cpu().numpy(),
        #                                        sampling_rate=self.target_sample_rate,
        #                                        return_tensors="pt")["input_values"].to(self.device)

        # x_l1_features = self.feature_extractor(x_l1.cpu().numpy(),
        #                                        sampling_rate=self.target_sample_rate,
        #                                        return_tensors="pt")["input_values"].to(self.device)

        # h1_inputs = self.processor(x_h1.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(self.device)
        # l1_inputs = self.processor(x_l1.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(self.device)
        # logits = self.classifier(h1_inputs, l1_inputs)
        # return logits

        x_h1_features = self.feature_extractor(x_h1.cpu().numpy(), sampling_rate=self.target_sample_rate, return_tensors="pt").to(x.device)
        if self.model_name == "wav2vec2-bert":
            x_h1_features = x_h1_features["input_features"].squeeze(0)
        else:
            x_h1_features = x_h1_features.input_values.squeeze(0)

        x_l1_features = self.feature_extractor(x_l1.cpu().numpy(), sampling_rate=self.target_sample_rate, return_tensors="pt").to(x.device)
        if self.model_name == "wav2vec2-bert":
            x_l1_features = x_l1_features["input_features"].squeeze(0)
        else:
            x_l1_features = x_l1_features.input_values.squeeze(0)
        
        logits = self.classifier(x_h1_features, x_l1_features)
        
        return logits

###########################################################################
# AST
###########################################################################

class ASTBinaryClassifier(nn.Module):
    def __init__(self, model, num_classes=1):
        super().__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, h1_inputs, l1_inputs):
        output_h1 = self.model(**h1_inputs).last_hidden_state.mean(dim=1)
        output_l1 = self.model(**l1_inputs).last_hidden_state.mean(dim=1)
        outputs = torch.cat([output_h1, output_l1], dim=-1)
        logits = self.classifier(outputs)
        return logits

class ASTModule(nn.Module):
    def __init__(
        self,
        ast_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        model_name: str = "AST",
        input_sample_rate: int = 2048,
        target_sample_rate: int = 16000,
        device: str = "cuda"
    ):
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.ast_model_name = ast_model_name
        self.model_name = model_name
        self.device = device

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(ast_model_name)
        self.ast_model = AutoModel.from_pretrained(ast_model_name)
        self.resampler = T.Resample(self.input_sample_rate, self.target_sample_rate)

        attn_modules = []
        for name, _ in self.ast_model.named_modules():
            if any(x in name for x in ["attn", "attention"]):
                if isinstance(_, nn.Linear):
                    attn_modules.append(name.split(".")[-1])

        lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=list(set(attn_modules)), bias="none")
        self.ast_model = get_peft_model(self.ast_model, lora_config)

        for n, p in self.ast_model.named_parameters():
            p.requires_grad = ("lora" in n)
        self.ast_model.print_trainable_parameters()

        self.classifier = ASTBinaryClassifier(self.ast_model).to(self.device)

    def forward(self, X):
        h1 = self.resampler(X[:, 0, :])
        l1 = self.resampler(X[:, 1, :])

        h1_features = self.feature_extractor(h1.cpu().numpy(), 
                                             sampling_rate=self.target_sample_rate, 
                                             return_tensors="pt").to(self.device)

        l1_features = self.feature_extractor(l1.cpu().numpy(), 
                                             sampling_rate=self.target_sample_rate, 
                                             return_tensors="pt").to(self.device)

        logits = self.classifier(h1_features, l1_features)
        return logits

###########################################################################
# WavLM
###########################################################################

class WavLMBinaryClassifier(nn.Module):
    def __init__(self, model, num_classes=1):
        super().__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size * 2, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, h1_input, l1_input):
        output_h1 = self.model(h1_input).last_hidden_state.mean(dim=1)
        output_l1 = self.model(l1_input).last_hidden_state.mean(dim=1)
        outputs = torch.cat((output_h1, output_l1), dim=1)
        logits = self.classifier(outputs)
        return logits

class WavLMModule(torch.nn.Module):
    def __init__(
        self,
        wavlm_model_name: str = "microsoft/wavlm-base",
        model_name: str = "WavLM",
        input_sample_rate: int = 2048,
        target_sample_rate: int = 16000,
        device: str = "cuda"
    ):
        super(WavLMModule, self).__init__()
        self.wavlm_model_name = wavlm_model_name
        self.model_name = model_name
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.device = device

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.wavlm_model_name) #"facebook/WavLM-large-960h"
        self.WavLM_model = AutoModel.from_pretrained(self.wavlm_model_name)
        self.resampler = T.Resample(orig_freq=self.input_sample_rate, new_freq=self.target_sample_rate)
        
        lin_names = [n for n, m in self.WavLM_model.named_modules() if isinstance(m, nn.Linear)]
        attn_lin  = [n for n in lin_names if ("attn" in n or "attention" in n)]
        cands = ["q_proj","k_proj","v_proj","out_proj","qkv","proj","query","key","value","out"]
        target_modules = [c for c in cands if any(c in n for n in attn_lin)] or \
                        sorted({n.split(".")[-1] for n in attn_lin}) or ["query","key","value","out"]
        print("LoRA target_modules =", target_modules)

        lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=target_modules)
        self.WavLM_model = get_peft_model(self.WavLM_model, lora_config).to(device)
        self.WavLM_model.print_trainable_parameters()
        
        # attn_modules = []
        # for name, _ in self.WavLM_model.named_modules():
        #     if any(x in name for x in ["attn", "attention"]):
        #         if isinstance(_, nn.Linear):
        #             attn_modules.append(name.split(".")[-1])
        
        # lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=32, target_modules=list(set(attn_modules)))
        
        # self.WavLM_model = get_peft_model(self.WavLM_model, lora_config)
        
        # # Freeze base parameters; only train LoRA parameters.
        # for name, param in self.WavLM_model.named_parameters():
        #     param.requires_grad = "lora" in name

        # self.WavLM_model.print_trainable_parameters()

        self.classifier = WavLMBinaryClassifier(self.WavLM_model).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract individual detector data
        if x.dim() == 3 and x.shape[1] == 2:
            x_h1 = x[:, 0, :]  # Channel 1
            x_l1 = x[:, 1, :]  # Channel 2
        else:
            raise ValueError("Input tensor must have shape [batch, 2, time].")
        
        # Resample to target sample rate
        x_h1 = self.resampler(x_h1)
        x_l1 = self.resampler(x_l1)

        x_h1_features = self.feature_extractor(x_h1.cpu().numpy(), sampling_rate=self.target_sample_rate, return_tensors="pt").to(x.device)
        x_h1_features = x_h1_features.input_values.squeeze(0)

        x_l1_features = self.feature_extractor(x_l1.cpu().numpy(), sampling_rate=self.target_sample_rate, return_tensors="pt").to(x.device)
        x_l1_features = x_l1_features.input_values.squeeze(0)
        
        logits = self.classifier(x_h1_features, x_l1_features)
        
        return logits
    
###########################################################################
# Hubert
###########################################################################

class HubertBinaryClassifier(nn.Module):
    def __init__(self, model, num_classes=1):
        super().__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size * 2, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, h1_input, l1_input):
        output_h1 = self.model(h1_input).last_hidden_state.mean(dim=1)
        output_l1 = self.model(l1_input).last_hidden_state.mean(dim=1)
        outputs = torch.cat((output_h1, output_l1), dim=1)
        logits = self.classifier(outputs)
        return logits

class HubertModule(torch.nn.Module):
    def __init__(
        self,
        hubert_model_name: str = "facebook/hubert-base-ls960",
        model_name: str = "hubert",
        input_sample_rate: int = 2048,
        target_sample_rate: int = 16000,
        device: str = "cuda"
    ):
        super(HubertModule, self).__init__()
        self.hubert_model_name = hubert_model_name
        self.model_name = model_name
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.device = device

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.hubert_model_name) #"facebook/hubert-large-960h"
        self.hubert_model = AutoModel.from_pretrained(self.hubert_model_name)
        self.resampler = T.Resample(orig_freq=self.input_sample_rate, new_freq=self.target_sample_rate)
        
        # lin_names = [n for n, m in self.hubert_model.named_modules() if isinstance(m, nn.Linear)]
        # attn_lin  = [n for n in lin_names if ("attn" in n or "attention" in n)]
        # cands = ["q_proj","k_proj","v_proj","out_proj","qkv","proj","query","key","value","out"]
        # target_modules = [c for c in cands if any(c in n for n in attn_lin)] or \
        #                 sorted({n.split(".")[-1] for n in attn_lin}) or ["query","key","value","out"]
        # print("LoRA target_modules =", target_modules)

        # lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=target_modules)
        # self.hubert_model = get_peft_model(self.hubert_model, lora_config).to(device)
        # self.hubert_model.print_trainable_parameters()
        
        attn_modules = []
        for name, _ in self.hubert_model.named_modules():
            if any(x in name for x in ["attn", "attention"]):
                if isinstance(_, nn.Linear):
                    attn_modules.append(name.split(".")[-1])
        
        lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=32, target_modules=list(set(attn_modules)))
        
        self.hubert_model = get_peft_model(self.hubert_model, lora_config)
        
        # Freeze base parameters; only train LoRA parameters.
        for name, param in self.hubert_model.named_parameters():
            param.requires_grad = "lora" in name

        self.hubert_model.print_trainable_parameters()

        self.classifier = HubertBinaryClassifier(self.hubert_model).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract individual detector data
        if x.dim() == 3 and x.shape[1] == 2:
            x_h1 = x[:, 0, :]  # Channel 1
            x_l1 = x[:, 1, :]  # Channel 2
        else:
            raise ValueError("Input tensor must have shape [batch, 2, time].")
        
        # Resample to target sample rate
        x_h1 = self.resampler(x_h1)
        x_l1 = self.resampler(x_l1)

        x_h1_features = self.feature_extractor(x_h1.cpu().numpy(), sampling_rate=self.target_sample_rate, return_tensors="pt").to(x.device)
        x_h1_features = x_h1_features.input_values.squeeze(0)

        x_l1_features = self.feature_extractor(x_l1.cpu().numpy(), sampling_rate=self.target_sample_rate, return_tensors="pt").to(x.device)
        x_l1_features = x_l1_features.input_values.squeeze(0)
        
        logits = self.classifier(x_h1_features, x_l1_features)
        
        return logits

###########################################################################
# Parakeet
###########################################################################

class ParakeetBinaryClassifier(nn.Module):
    def __init__(self, model, num_classes=1):
        super().__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size * 2, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, h1_input, l1_input):
        output_h1 = self.model(h1_input).last_hidden_state.mean(dim=1)
        output_l1 = self.model(l1_input).last_hidden_state.mean(dim=1)
        outputs = torch.cat((output_h1, output_l1), dim=1)
        logits = self.classifier(outputs)
        return logits

class ParakeetModule(torch.nn.Module):
    def __init__(
        self,
        parakeet_model_name: str = "nvidia/parakeet-ctc-1.1b",
        model_name: str = "parakeet",
        input_sample_rate: int = 2048,
        target_sample_rate: int = 16000,
        device: str = "cuda"
    ):
        super(ParakeetModule, self).__init__()
        self.parakeet_model_name = parakeet_model_name
        self.model_name = model_name
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.device = device

        self.parakeet_model = AutoModel.from_pretrained(self.parakeet_model_name)
        self.resampler = T.Resample(orig_freq=self.input_sample_rate, new_freq=self.target_sample_rate)
        
        attn_modules = []
        for name, _ in self.parakeet_model.named_modules():
            if any(x in name for x in ["attn", "attention"]):
                if isinstance(_, nn.Linear):
                    attn_modules.append(name.split(".")[-1])
        
        lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=32, target_modules=list(set(attn_modules)))
        
        self.parakeet_model = get_peft_model(self.parakeet_model, lora_config)
        
        # Freeze base parameters; only train LoRA parameters.
        for name, param in self.parakeet_model.named_parameters():
            param.requires_grad = "lora" in name

        self.parakeet_model.print_trainable_parameters()

        self.classifier = ParakeetBinaryClassifier(self.parakeet_model).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract individual detector data
        if x.dim() == 3 and x.shape[1] == 2:
            x_h1 = x[:, 0, :]  # Channel 1
            x_l1 = x[:, 1, :]  # Channel 2
        else:
            raise ValueError("Input tensor must have shape [batch, 2, time].")
        
        # Resample to target sample rate
        x_h1 = self.resampler(x_h1)
        x_l1 = self.resampler(x_l1)

        x_h1 = x_h1 / (x_h1.abs().max(dim=1, keepdim=True).values + 1e-8)
        x_l1 = x_l1 / (x_l1.abs().max(dim=1, keepdim=True).values + 1e-8)

        # h1_inputs = self.processor(x_h1.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(self.device)
        # l1_inputs = self.processor(x_l1.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(self.device)
        logits = self.classifier(x_h1, x_l1)

        # x_h1_features = self.feature_extractor(x_h1.cpu().numpy(), sampling_rate=self.target_sample_rate, return_tensors="pt").to(x.device)
        # x_h1_features = x_h1_features.input_values.squeeze(0)

        # x_l1_features = self.feature_extractor(x_l1.cpu().numpy(), sampling_rate=self.target_sample_rate, return_tensors="pt").to(x.device)
        # x_l1_features = x_l1_features.input_values.squeeze(0)
        
        # logits = self.classifier(x_h1_features, x_l1_features)
        
        return logits
    
###########################################################################
# Mimi
###########################################################################

class MimiBinaryClassifier(nn.Module):
    def __init__(self, model, num_classes=1):
        super().__init__()
        self.model = model
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size * 2, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, h1_emb, l1_emb):
        output_h1 = h1_emb.mean(dim=-1)
        output_l1 = l1_emb.mean(dim=-1)
        outputs = torch.cat((output_h1, output_l1), dim=1)
        logits = self.classifier(outputs)
        return logits

class MimiModule(torch.nn.Module):
    def __init__(
        self,
        mimi_model_name: str = "kyutai/mimi",
        model_name: str = "mimi",
        input_sample_rate: int = 2048,
        target_sample_rate: int = 24000,
        device: str = "cuda"
    ):
        super(MimiModule, self).__init__()
        self.mimi_model_name = mimi_model_name
        self.model_name = model_name
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.device = device

        # self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.mimi_model_name)
        self.processor = AutoProcessor.from_pretrained(self.mimi_model_name)
        self.mimi_model = AutoModel.from_pretrained(self.mimi_model_name)
        self.resampler = T.Resample(orig_freq=self.input_sample_rate, new_freq=self.target_sample_rate)
        
        attn_modules = []
        for name, _ in self.mimi_model.named_modules():
            if any(x in name for x in ["attn", "attention"]):
                if isinstance(_, nn.Linear):
                    attn_modules.append(name.split(".")[-1])

        lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=32, target_modules=list(set(attn_modules)))#matched_modules)
        
        self.mimi_model = get_peft_model(self.mimi_model, lora_config)
        
        # Freeze base parameters; only train LoRA parameters.
        for name, param in self.mimi_model.named_parameters():
            param.requires_grad = "lora" in name

        self.mimi_model.print_trainable_parameters()

        self.classifier = MimiBinaryClassifier(self.mimi_model).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract individual detector data
        if x.dim() == 3 and x.shape[1] == 2:
            x_h1 = x[:, 0, :]  # Channel 1
            x_l1 = x[:, 1, :]  # Channel 2
        else:
            raise ValueError("Input tensor must have shape [batch, 2, time].")
        
        # Resample to target sample rate
        x_h1 = self.resampler(x_h1)
        x_l1 = self.resampler(x_l1)

        h1_np = [a for a in x_h1.cpu().numpy()]
        l1_np = [a for a in x_l1.cpu().numpy()]

        h1_features = self.processor(h1_np, sampling_rate=self.target_sample_rate, return_tensors="pt", padding=True).input_values.to(self.device)
        l1_features = self.processor(l1_np, sampling_rate=self.target_sample_rate, return_tensors="pt", padding=True).input_values.to(self.device)
        
        h1_codes = self.mimi_model.encode(h1_features).audio_codes
        l1_codes = self.mimi_model.encode(l1_features).audio_codes

        emb_h1 = self.mimi_model.quantizer.decode(h1_codes)
        emb_l1 = self.mimi_model.quantizer.decode(l1_codes)

        # Classifier expects embeddings
        logits = self.classifier(emb_h1, emb_l1)

        return logits