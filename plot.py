import matplotlib as plt
import numpy as np
import pandas as pd
from PIL import Image

IMAGE_FILE_PATH = '{image_path}'
DATA_PATH = '{data_path}'

class drawPlot():
    # 사진 위에 키포인트 출력
    def showKeypoints():
        plt.figure(figsize=(40,20))
        count=1

        for i in range(len(IMAGE_FILE_PATH)):
            
            plt.subplot(3,1, count)
            img_sample_path = IMAGE_FILE_PATH[i]
            img = Image.open(img_sample_path)
            img_np = np.array(img)

            keypoint = pd.DataFrame(DATA_PATH)
            keypoint = keypoint.iloc[:,1:35] #위치 키포인트 하나씩 확인
            keypoint_sample = keypoint.iloc[i, :]
            
            for j in range(0,len(keypoint.columns),2):
                plt.plot(keypoint_sample[j], keypoint_sample[j+1],'rx')
                plt.imshow(img_np)
            count += 1

    def run(self):
        self.showKeypoints()

if __name__ == "__main__":
    my_plot = drawPlot()
    my_plot.run()
