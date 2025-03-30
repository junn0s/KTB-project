# Fruits & Vegetables Image Classification with CNNs

**A comparative study of five CNN architectures (SimpleCNN, VGG16, ResNet50, MobileNetV2, ShuffleNetV2) for efficient multi-class classification of fruit and vegetable images.**

---

## Overview

이 프로젝트는 **36가지 과일 및 채소 이미지 분류**를 목표로 다양한 CNN 기반 모델의 성능을 비교하고 효율적인 모델을 선정하는 개인 연구 프로젝트입니다

- **목표**: 식재료 자동 인식 기능을 통해 요리 레시피 추천 앱에서의 사용자 입력을 최소화
- **데이터셋**: [Kaggle - Fruits and Vegetables Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) (총 3,825장)
- **클래스 수**: 36종
- **주요 모델**: SimpleCNN, VGG16, ResNet50, MobileNetV2, ShuffleNetV2

---

## Models Compared

| Model           | Params (M) | FLOPs (M) | Test Accuracy (%) | 특징 |
|----------------|------------|-----------|--------------------|------|
| SimpleCNN       | 4.0        | 185       | 95.54              | 경량 구조, 학습 빠름 |
| VGG16           | 138.0      | 15,300    | 96.38              | 고성능, 연산량 높음 |
| ResNet50        | 25.6       | 4,100     | 94.15              | 깊은 구조, 데이터 증강 필요 |
| MobileNetV2     | 3.4        | 465       | **97.21**          | 최고 성능, 경량화 우수 |
| ShuffleNetV2    | **2.3**    | **62**    | 96.66              | 가장 적은 파라미터, 모바일 최적 |

>  **최고 성능 모델**: MobileNetV2  
>  **최고 효율 모델**: ShuffleNetV2

---

##  Tech Stack

- **Python 3.12**
- **PyTorch**
- **Google Colab (A100 GPU)**
- **Streamlit**: 학습 시각화 대시보드
- **Aiven MySQL + DBeaver**: 학습 로그 저장
- **Kaggle API**: 데이터 다운로드

---

##  Experiment Details

- 이미지 크기: 180×180 ~ 224×224
- 배치 사이즈: 32 ~ 64
- Epoch 수: 12~50 (모델별 상이, EarlyStopping 적용)
- 정규화, 데이터 증강, 드롭아웃, Adam 옵티마이저 사용

---

##  Visualization

각 모델의 학습 정확도 및 손실, 검증 정확도는 Streamlit 대시보드로 시각화하였습니다

![모델 성능 시각화](/images/mobilenet_train_val_acc_db.png)  
<sub>Figure: MobileNetV2</sub>

---

##  Project Structure

```
📁 KTB-project/
├── README.md
├── .gitignore
├── download_kaggle_data.py
├── mysql_table.sql
├── streamlit_cnn.py
├── MobileNet_V2.py
├── ResNet50.py
├── ShuffleNet_0.5.py
├── VGG16.py
├── simpleCNN.py
├── images/
│   └── mobilenet_train_val_acc_db.png
├── fruit-and-vegetable-image/
│   ├── test/
│   └── train/
│   └── validation/
└── 개인 프로젝트 보고서(milo.park).pdf
```

---

##  Reference

- [Kaggle Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
- [Papers with Code - ShuffleNet](https://paperswithcode.com/method/shufflenet)
- [Papers with Code - MobileNetV2](https://paperswithcode.com/method/mobilenetv2)
- [VGG / ResNet](https://paperswithcode.com/method/resnet)

---

>  Developed by milo | March 2025
