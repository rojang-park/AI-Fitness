# 2021 AI-Fitness Project
Project nickname : AI 헬스 코치

Project execution period : 2021.07~2021.08

## Tech Stack
- 이미지 속 인물의 신체부위 검출을 위해 kakao pose  API 사용
- 다양한 운동 동작 학습을 위해 20만 건의 이미지 및 json 데이터 처리
- 시계열 데이터 처리를 위한 LSTM, CNN 알고리즘 사용
- openCV, matplotlib 을 이용한 인식 결과 시각화


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

`inference.py` inference data

`main.py` Main
