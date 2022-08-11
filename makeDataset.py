import pandas as pd
import glob
import matplotlib.pyplot as plt
import requests
from matplotlib import image as mpimg
from tqdm import tqdm, tqdm_notebook

img_path = glob.glob('C:/Users/KSP/Downloads/피트니스 자세 이미지/Training/Day29_201031_F/A/*/*/*/*.jpg')
keypoints = ['nose_x','nose_y','nose_score','left_eye_x', 'left_eye_y', 'left_eye_score','right_eye_x','right_eye_y', 'right_eye_score',
                               'left_ear_x', 'left_ear_y', 'left_ear_scre','right_ear_x', 'right_ear_y', 'right_ear_score','left_shoulder_x', 'left_shoulder_y', 'left_shoulder_score',
                               'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_score','left_elbow_x', 'left_elbow_y', 'left_elbow_score',
                               'right_elbow_x', 'right_elbow_y', 'right_elbow_score','left_wrist_x', 'left_wrist_y', 'left_wrist_score','right_wrist_x', 'right_wrist_y', 'right_wrist_score','left_hip_x', 'left_hip_y', 'left_hip_score',
                               'right_hip_x', 'right_hip_y', 'right_hip_score','left_knee_x', 'left_knee_y', 'left_knee_score','right_knee_x', 'right_knee_y', 'right_knee_score',
                               'left_ankle_x', 'left_ankle_y', 'left_ankle_score','right_ankle_x', 'right_ankle_y', 'left_ankle_score']

#카카오 API 설정
APP_KEY = '[API KEY]'
session = requests.Session()
session.headers.update({'Authorization': 'KakaoAK ' + APP_KEY})

# api로 이미지 키포인트 받기
def getKeypoints():

    response_json = []

    for i in tqdm_notebook(range(len(IMAGE_FILE_PATH))):
        img = open(IMAGE_FILE_PATH[i], 'rb')
        response = session.post('https://cv-api.kakaobrain.com/pose', files=[('file', img)])
        response.raise_for_status() 
        print(response.status_code, response.json())
        response_json.append(response.json())
        print(i)

# json에서 필요한 데이터 추출
def dismissJSON():
    for i in range(0,23040):
        keypoints.append(response_json[i][0]['keypoints'])

def divideInList():
    n = 1 # 한 리스트에 몇개씩 담을지 결정
    result = [keypoints[0][i * n:(i + 1) * n] for i in range((len(keypoints[0]) + n - 1) // n )]

    self.arr = []
    number = 0

    for keypoint in keypoints:

    
        result = [keypoints[number][i * n:(i + 1) * n] for i in range((len(keypoints[number]) + n - 1) // n)]
        self.arr.append(result)
        number = number + 1

#df 편집
def editDataframe():
    df = pd.DataFrame(self.arr, columns = keypoints)

    # 불필요 컬럼 삭제
    df.drop([df.columns[2], df.columns[5], df.columns[8], df.columns[11], df.columns[14],
                df.columns[17], df.columns[20], df.columns[23], df.columns[26],
                df.columns[29], df.columns[32], df.columns[35], df.columns[38],
                df.columns[41], df.columns[44], df.columns[47], df.columns[50]], axis = 1, inplace = True)

    #
    train_filename = []

    for t_paths in tqdm(IMAGE_FILE_PATH):
        filename = t_paths.split('\\')[-1]
        train_filename.append(filename)

    df['path'] = IMAGE_FILE_PATH
    df['image'] = train_filename

    #컬럼 순서 변경
    df = df[['image', 'nose_x', 'nose_y', 'left_eye_x', 'left_eye_y', 'right_eye_x',
        'right_eye_y', 'left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y',
        'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x',
        'right_shoulder_y', 'left_elbow_x', 'left_elbow_y', 'right_elbow_x',
        'right_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x',
        'right_wrist_y', 'left_hip_x', 'left_hip_y', 'right_hip_x',
        'right_hip_y', 'left_knee_x', 'left_knee_y', 'right_knee_x',
        'right_knee_y', 'left_ankle_x', 'left_ankle_y', 'right_ankle_x',
        'right_ankle_y', 'path']]

    #
    df.to_csv('data.csv', index = False)
