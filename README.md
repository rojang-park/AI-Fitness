# 2021 AI-Fitness Project
## 피트니스 자세 인식 및 피드백 시스템
Project nickname : AI 헬스 코치
Project execution period : 2021.07~2021.08


## Description
AI 헬스 코치는 운동 이미지를 입력받아 인공지능으로 분석해 자세를 인식하고 피드백을 제공한다.

![image](https://user-images.githubusercontent.com/109723552/183353440-5727e163-3ac5-4317-ad28-8132a0c619d7.png)

### 1. function list

|구분|기능|구현|
|:---|:---|:---|
|S/W|키포인트 추출|OPEN API(KAKAO)|
|S/W|이미지 분석|Deep Learning|
|S/W|시각화|MatplotLib|

## Files

`makeDataset.py` detect keypoint from image & make dataset

`train.py` train AI model

`plot.py` draw result on image

`main.py` Main
