from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil
import os
import sys
# from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import torch.nn.functional as F

# Вспомогательная функция для скачивания модели с Hugging Face Hub
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)

# Вспомогательная функция для загрузки модели из локального пути
def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model

# Вспомогательная функция для загрузки модели по repo_id
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)

if __name__ == '__main__':
    # Загрузка модели
    HF_TOKEN = 'token'  # Замените на ваш токен, если требуется
    path = os.path.expanduser('~/.cvlface_cache/minchul/cvlface_DFA_resnet50')  # Путь для сохранения модели
    repo_id = 'minchul/cvlface_DFA_resnet50'  # Идентификатор модели на Hugging Face
    aligner = load_model_by_repo_id(repo_id, path, HF_TOKEN)



from transformers import PreTrainedModel
from transformers import PretrainedConfig
from omegaconf import OmegaConf
from .aligners import get_aligner
import yaml

class ModelConfig(PretrainedConfig):

    def __init__(
            self, path='aligner/pretrained_model/model.yaml',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.conf = dict(yaml.safe_load(open(path)))


class CVLFaceAlignmentModel(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, cfg, path='aligner/pretrained_model/model.pt'):
        super().__init__(cfg)
        model_conf = OmegaConf.create(cfg.conf)
        self.model = get_aligner(model_conf)
        self.model.load_state_dict_from_path(path)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)