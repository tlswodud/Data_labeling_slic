# Data_labeling_slic

고화질 드론 이미지 tif 형식을 100 * 100 pixel로  분할하고

SLIC (super pixel algolithm) 을 사용하여

비슷한 색상의 이미지에 맞게 분할했습니다.

형성된 segement는 마우스 이벤트를 받아 

background / diseased leaf / healty leaf 로 라벨링 하고 

인공신경망 학습을 위한 데이터 형식에 맞게 Pytoch tensor 로 변환하였습니다.

구동 이미지 화면 및 segement 이미지



![image](https://github.com/user-attachments/assets/a58a6368-fb66-4630-b3ca-503c78152536)
