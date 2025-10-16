# GPU Resource Monitoring for VERL Training

이 문서는 VERL 훈련 중 GPU 리소스 사용량을 실시간으로 모니터링하고 WandB에서 시각화하는 방법을 설명합니다.

verl 0.5.0 버전으로 개발되었으며, 설치 시작 시에는 [![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/) 를 참고하세요. 

## 빠른 시작

### 1. 필수 패키지 설치

```bash
pip install pynvml psutil wandb
```

### 2. GRPO 훈련에서 GPU 모니터링 활성화

기존 GRPO 훈련 스크립트에 다음 옵션을 추가하세요:

```bash
python3 -m verl.trainer.main_ppo \
    # ... 기존 설정들 ... \
    +trainer.enable_gpu_monitoring=true \
    +trainer.gpu_monitor_interval=2.0
```

## 모니터링되는 메트릭

### GPU 메트릭 (각 GPU별)
- **GPU 사용률** (`gpu_N_gpu_utilization_pct`): GPU 코어 사용률 (%)
- **메모리 사용률** (`gpu_N_memory_utilization_pct`): GPU 메모리 사용률 (%)
- **메모리 할당량** (`gpu_N_memory_allocated_gb`): PyTorch가 할당한 메모리 (GB)
- **메모리 예약량** (`gpu_N_memory_reserved_gb`): PyTorch가 예약한 메모리 (GB)
- **총 메모리** (`gpu_N_memory_total_gb`): GPU 총 메모리 (GB)
- **온도** (`gpu_N_temperature_c`): GPU 온도 (°C)
- **전력 사용량** (`gpu_N_power_usage_w`): 전력 소비량 (W)
- **클럭 속도** (`gpu_N_graphics_clock_mhz`, `gpu_N_memory_clock_mhz`): GPU/메모리 클럭 (MHz)

### 시스템 메트릭
- **CPU 사용률** (`cpu_utilization_pct`): CPU 사용률 (%)
- **시스템 메모리** (`system_memory_used_gb`, `system_memory_utilization_pct`): 시스템 RAM 사용량

## 사용 방법

### 방법 1: 훈련과 통합된 모니터링 (권장)

GRPO 훈련 스크립트에 GPU 모니터링을 통합:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    # ... 기타 설정 ... \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='your_project' \
    trainer.experiment_name='your_experiment' \
    +trainer.enable_gpu_monitoring=true \
    +trainer.gpu_monitor_interval=1.0
```

**설정 옵션:**
- `trainer.enable_gpu_monitoring`: GPU 모니터링 활성화 (기본값: false)
- `trainer.gpu_monitor_interval`: 모니터링 간격 (초, 기본값: 1.0)

### 방법 2: 독립적인 GPU 모니터링

훈련과 별도로 GPU 모니터링만 실행:

```bash
# 기본 모니터링 (콘솔 출력만)
python scripts/monitor_gpu_realtime.py

# WandB와 함께 모니터링
python scripts/monitor_gpu_realtime.py \
    --wandb-project "gpu_monitoring" \
    --wandb-run-name "training_session_1" \
    --interval 0.5

# 특정 GPU만 모니터링
python scripts/monitor_gpu_realtime.py \
    --gpus "0,1,2,3" \
    --interval 2.0 \
    --wandb-project "gpu_monitoring"

# 콘솔 출력 없이 WandB만
python scripts/monitor_gpu_realtime.py \
    --wandb-project "gpu_monitoring" \
    --no-console
```

**독립 모니터링 옵션:**
- `--interval`: 모니터링 간격 (초)
- `--wandb-project`: WandB 프로젝트 이름
- `--wandb-run-name`: WandB 실행 이름
- `--gpus`: 모니터링할 GPU ID (쉼표로 구분)
- `--no-console`: 콘솔 출력 비활성화
- `--summary-interval`: 요약 출력 간격 (초)

## WandB에서 시각화

WandB 대시보드에서 다음과 같은 차트를 만들 수 있습니다:

### 1. GPU 사용률 차트
```
Y축: gpu_0_gpu_utilization_pct, gpu_1_gpu_utilization_pct, ...
X축: Step
```

### 2. GPU 메모리 사용량 차트
```
Y축: gpu_0_memory_allocated_gb, gpu_0_memory_total_gb
X축: Step
```

### 3. GPU 온도 모니터링
```
Y축: gpu_0_temperature_c, gpu_1_temperature_c, ...
X축: Step
```

### 4. 전력 소비량 추적
```
Y축: gpu_0_power_usage_w, gpu_1_power_usage_w, ...
X축: Step
```

## 고급 사용법

### 커스텀 모니터링 구현

```python
from verl.utils.gpu_monitor import GPUResourceMonitor

# 커스텀 모니터 생성
monitor = GPUResourceMonitor(
    monitor_interval=0.5,  # 0.5초마다 모니터링
    log_to_wandb=True,
    log_to_console=True,
    device_ids=[0, 1, 2, 3]  # 특정 GPU만 모니터링
)

# 모니터링 시작
monitor.start_monitoring()

# 최신 메트릭 가져오기
latest_metrics = monitor.get_latest_metrics()
print(f"GPU 0 사용률: {latest_metrics['gpu_0_gpu_utilization_pct']:.1f}%")

# 모니터링 중지
monitor.stop_monitoring()
```

### 기존 코드에 통합

```python
from verl.utils.tracking import Tracking

# GPU 모니터링이 활성화된 Tracking 인스턴스 생성
logger = Tracking(
    project_name="my_project",
    experiment_name="my_experiment", 
    default_backend=["wandb"],
    enable_gpu_monitoring=True,
    gpu_monitor_interval=1.0
)

# 이후 logger.log() 호출 시 GPU 메트릭도 자동으로 로깅됨
```

## 주의사항

1. **성능 영향**: 모니터링 간격이 너무 짧으면 (< 0.5초) 훈련 성능에 영향을 줄 수 있습니다.

2. **메모리 사용량**: 장시간 모니터링 시 메트릭 히스토리가 메모리를 사용합니다.

3. **권한**: NVML 기능을 사용하려면 적절한 GPU 드라이버와 권한이 필요합니다.

## 문제 해결

### 일반적인 문제들

1. **NVML 초기화 실패**
   ```bash
   pip install pynvml
   # 또는 NVIDIA 드라이버 업데이트
   ```

2. **WandB 로그인 문제**
   ```bash
   wandb login
   ```

3. **권한 문제**
   ```bash
   # Docker 환경에서는 --gpus all 옵션 필요
   docker run --gpus all ...
   ```

4. **메모리 부족**
   - 모니터링 간격을 늘리거나 (`gpu_monitor_interval` 증가)
   - 메트릭 히스토리 제한 설정

## 예제 출력

콘솔 출력 예제:
```
Starting GPU monitoring...
Monitoring interval: 1.0s
Monitoring GPUs: [0, 1, 2, 3, 4, 5, 6, 7]
WandB logging: ✓
Console output: ✓

GPU Monitor - GPU0: 85.2%/76.3% | GPU1: 82.1%/74.8% | GPU2: 88.5%/78.2% | GPU3: 84.7%/75.9%
```

WandB 대시보드에서는 실시간으로 업데이트되는 차트들을 볼 수 있습니다.
