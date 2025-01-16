import torch
import os
import shutil
import importlib.util

EXCEPTION_FOLDERS = [
    'checkpoints',
    'inputs',
    'outputs',
    'wandb',
    'debug',
    '__pycache__',

]
EXCEPTION_EXTENSIONS = [
    '.pth',
    '.png',
    '.jpg',
    '.jpeg',
    '.mp4',
    '.pyc',
]

def get_layer_wise_learning_rates(model):  # Hardcoded learning rates for each layer
    print("Layer-wise learning rates are hardcoded now. TODO: Make it configurable")
    params = []

    # DDP로 래핑된 모델의 경우 원래 모델을 model.module로 접근
    actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # Layer별 learning rate 설정
    params.append({'params': actual_model.dense_feature_encoder.parameters(), 'lr': 1e-4})
    params.append({'params': actual_model.latent_token_encoder.parameters(), 'lr': 5e-5})
    params.append({'params': actual_model.latent_token_decoder.parameters(), 'lr': 5e-5})
    params.append({'params': actual_model.implicit_motion_alignment.parameters(), 'lr': 1e-4})
    params.append({'params': actual_model.frame_decoder.parameters(), 'lr': 2e-4})
    # params.append({'params': actual_model.mapping_network.parameters(), 'lr': 1e-4})

    return params

# helper for gradient vanishing / explosion
def hook_fn(name):
    def hook(grad):
        if torch.isnan(grad).any():
            # print(f"🔥 NaN gradient detected in {name}")
            return torch.zeros_like(grad)  # Replace NaN with zero
        elif torch.isinf(grad).any():
            # print(f"🔥 Inf gradient detected in {name}")
            return torch.clamp(grad, -1e6, 1e6)  # Clamp infinite values
        #else:
            # You can add more conditions or logging here
         #  grad_norm = grad.norm().item()
         #   print(f"Gradient norm for {name}: {grad_norm}")
        return grad
    return hook

def add_gradient_hooks(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(hook_fn(name))

def find_package_path(package_name):
    # 패키지 로더 가져오기
    package_loader = importlib.util.find_spec(package_name)
    if package_loader is None or not hasattr(package_loader, 'submodule_search_locations'):
        raise ImportError(f"패키지 '{package_name}'의 경로를 찾을 수 없습니다.")
    
    # NamespaceLoader의 _path에서 첫 번째 경로 선택
    namespace_path = list(package_loader.submodule_search_locations)
    if not namespace_path:
        raise ImportError(f"패키지 '{package_name}'의 경로를 찾을 수 없습니다.")
    
    # 첫 번째 경로를 절대 경로로 반환
    return os.path.abspath(namespace_path[0])

def copy_all_files(src, dst):
    # copy all files from src to dst
    # except EXCEPTION_FOLDERS and EXCEPTION_EXTENSIONS
    if not os.path.exists(dst):
        os.makedirs(dst)

    for root, dirs, files in os.walk(src):
        # 제외할 폴더 필터링
        dirs[:] = [d for d in dirs if d not in EXCEPTION_FOLDERS]
        
        # 파일 복사
        for file in files:
            # 제외할 확장자 확인
            if any(file.endswith(ext) for ext in EXCEPTION_EXTENSIONS):
                continue
            
            src_file = os.path.join(root, file)
            # 대상 디렉토리 계산
            relative_path = os.path.relpath(root, src)
            dest_dir = os.path.join(dst, relative_path)
            
            # 대상 디렉토리 생성
            os.makedirs(dest_dir, exist_ok=True)
            
            # 파일 복사
            dest_file = os.path.join(dest_dir, file)
            shutil.copy2(src_file, dest_file)


if __name__ == "__main__":
    print(find_package_path('IMF_talking'))  # 패키지 경로 출력
    copy_all_files(find_package_path('IMF_talking'), '/mnt/CINELINGO_BACKUP/mingi/anycode/IMF/IMF_talking/IMF_talking/outputs/IMF_talking_default_multi4_zero_one/codes')  # 모든 파일 복사