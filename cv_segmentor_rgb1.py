from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import os
from PIL import Image
import numpy as np 
import spectral as spec
import shutil
import cv2 
import pickle


map_step = 2000
patch_size = 300
rgb_band = [2, 1, 0]

class Segmentor():
    #초기화 메서드 
    def __init__(self, source, ahsi, save_dir, y, x, block_num, init_folder=False):
        super().__init__()

        print("image blocks :", y, x)
        print("block num :", block_num)

        self.y = y
        self.x = x
        self.source = source #이미지 디렉토리
        self.ahsi = ahsi
        self.aRGBi = self._generateRGB(ahsi, y, x, rgb=True)
        self.aRGBi_m = self._generateRGB(ahsi, y, x, rgb=True)
        self.segments = self._generateSegments(self.aRGBi)
        self.segment_num = np.amax(self.segments)
        self.save_dir = save_dir
        self.block_num = block_num

        self.patch_save_dir = self.source + "PAT2\\" + self.save_dir
        self.num_of_class = 3
        self.classess = ["diseased", "leaf", "background"]
        self.selected_labels = [[] for _ in range(3)]
        self.colors = [[76, 76, 255], [0, 255, 0], [188, 102, 150]]
        self.active_class = 0
        self.extensions = ["jpg"]

        self.mode = "observe"
        self.lbtn = False

        if init_folder:
            self._initFolders()

    def _generateRGB(self, ahsi, y, x, rgb=False): #RGB 이미지 생성
        if rgb:
            region = ahsi.crop(x, y, map_step, map_step)
            region = region.numpy()
            region = region[:, :, rgb_band]
            print(region[0, 0, :])
            return region.astype(np.uint8)
        else:
            aRGBi = ahsi[y:y+map_step, x:x+map_step, rgb_band]
            return (aRGBi * 255 / np.max(aRGBi)).astype(np.uint8)

    def _generateSegments(self, aRGBi): #세분화 
        small_rgbi = cv2.resize(aRGBi, (100, 100)) #  전체 

        #slic 알고리즘 
        
        
        segments = slic(cv2.resize(small_rgbi, (self.aRGBi.shape[1], self.aRGBi.shape[0])),  
                        n_segments = 300, sigma = 30)
        rys = np.array([])
        rxs = np.array([])
        for s in range(1, np.amax(segments) + 1):
                c = np.where(segments == s)
                
                y_loc = [np.amin(c[0]), np.amax(c[0])]
                x_loc = [np.amin(c[1]), np.amax(c[1])]

                ry = y_loc[1] - y_loc[0] 
                rx = x_loc[1] - x_loc[0]

                rys = np.append(rys, ry)
                rxs = np.append(rxs, rx)

        print(np.average(rys), np.amax(rys), np.amin(rys))
        print(np.average(rxs), np.amax(rxs), np.amin(rxs))

        return segments

    def _patchBuilder(self, label):
        c = np.where(self.segments == label)
        
        y_loc = [np.amin(c[0]), np.amax(c[0])]
        x_loc = [np.amin(c[1]), np.amax(c[1])]

        ry = y_loc[1] - y_loc[0] 
        rx = x_loc[1] - x_loc[0]

        if ry < patch_size and rx < patch_size:
            mat = np.zeros((ry, rx, 3), dtype=np.uint8)
            for iy, my in zip(range(y_loc[0], y_loc[1]), range(ry)):
                for ix, mx in zip(range(x_loc[0], x_loc[1]), range(rx)):
                    if self.segments[iy][ix] == label:
                        mat[my][mx] = self.aRGBi_m[iy, ix, :]

            dy = int((patch_size - ry) / 2)
            dx = int((patch_size - rx) / 2)

            mat2 = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            mat2[dy+1:dy+ry+1, dx+1:dx+rx+1, :] = mat[:, :, :]

            return mat2
        else:
            return None

    def _initFolders(self):
        if os.path.isdir(self.source + "\\PAT2"):
            shutil.rmtree(self.source + "\\PAT2")

        os.mkdir(self.source + "\\PAT2")
        os.mkdir(self.patch_save_dir)
        os.mkdir(self.patch_save_dir + "\\SEGMENTS")
        for ex in self.extensions:
            os.mkdir(self.patch_save_dir + "\\" + ex)
            for c in self.classess:
                os.mkdir(self.patch_save_dir + "\\" + ex + "\\" + c)

    def _saveSegmentsandDisplay(self): # 저장 폴더 지정
        ssdir = f'{self.patch_save_dir}\\SEGMENTS\\segment_{self.block_num}_'
        with open(ssdir + "segment_labels.pckl", "rb") as f:
            pickle.dump(self.selected_labels, f)
        np.save(ssdir + "segments.npy", self.segments)
        cv2.imwrite(ssdir + "segmentation.jpg", cv2.resize((mark_boundaries(self.aRGBi, self.segments) * 255).astype(np.uint8), (900,900)))
        cv2.imwrite(ssdir + "img_classified.jpg", cv2.resize(self.aRGBi, (900,900)))
        cv2.imwrite(ssdir + "img.jpg", cv2.resize(self.aRGBi_m, (900,900)))
     
    def _saveBtn(self):
        print("Saving segments . . .")
        #self._saveSegmentsandDisplay()
        print("Saving patches . . .")
        for c in range(len(self.selected_labels)):
            for label in self.selected_labels[c]:
                patches = self._patchBuilder(label)
                if patches is not None:
                    patch = patches
                    for ex in self.extensions:
                        if ex == "jpg":
                            self._saveJPG(patch, self.patch_save_dir, self.classess[c], label)
        else:
            print("done")

    def _saveJPG(self, img, path, c, label): 
        PIL_img = Image.fromarray(img) #pilow 형식으로 변환 
        PIL_img.save(f'{path}\\jpg\\{c}\\{self.block_num}_{label}.jpg') # jpg 파일로 저장

    def printSelected(self): #선택한 세그먼트의 총개수를 보여줌
        print("Number of segments:", self.segment_num) #세그먼트 수
        total = 0
        for i in range(self.num_of_class):
            len_ = len(self.selected_labels[i])
            print(f'Notation: {i}, Num: {len_}, {self.classess[i]}')
            total += len_
        print("hTotal classified segments: ", total)
        print(f'Unclassified segments: {self.segment_num - total} left \n')

    def colorize(self, label):#선택한 것의 세그먼트의 색상을 다르게 함
        c = np.where(self.segments == label)
        self.aRGBi[c[0], c[1]] = self.colors[self.active_class]

    def colorizeRest(self): # 나머지 부분 보통 배경 부분을 한번에 처리하기 위해서 작성
        print("Colorizing . . .")
        for label in range(1, self.segment_num + 1):
            if label not in self.selected_labels[self.active_class]:
                for i in range(len(self.selected_labels)):
                    if i != self.active_class and label in self.selected_labels[i]:
                        break
                else:
                    self.selected_labels[self.active_class].append(label)
                    self.colorize(label)
        
        self.printSelected()

    def decolorize(self, label):
        c = np.where(self.segments == label)
        self.aRGBi[c[0], c[1]] = self.aRGBi_m[c[0], c[1]]
    
    def click_event(self, event, x, y, flags, params): #cv2 를 통해 마우스 이벤트로 선택
     
        label = self.segments[int(y/params)][int(x/params)]

        if event == cv2.EVENT_LBUTTONUP and self.lbtn:
            self.lbtn = False

        elif event == cv2.EVENT_LBUTTONDOWN: # 버튼 눌렸을때 잘못선택한거라면 제거해주고 아니라면 추가
            self.lbtn = True
            if self.mode == "draw":
                print("Region num: ", label)
                for sl in self.selected_labels:
                    if label in sl:         
                        sl.remove(label)
                        self.printSelected()
                        self.decolorize(label)
                        break
                else:
                    self.selected_labels[self.active_class].append(label)
                    self.printSelected()
                    self.colorize(label)

            if self.mode == "observe":
                print(x, ' ', y)
                print(self.active_class)

                

        elif event == cv2.EVENT_MOUSEMOVE and self.lbtn:
            if self.mode == "draw":
                for i in range(len(self.selected_labels)):
                    if i != self.active_class and label in self.selected_labels[i]:
                        self.selected_labels[i].remove(label)
                        self.decolorize(label)
                        self.colorize(label)
                        self.printSelected()

                    elif i == self.active_class and label not in self.selected_labels[i]:
                        self.selected_labels[i].append(label)
                        self.colorize(label)
                        self.printSelected()

            if self.mode == "observe":
                print(x, ' ', y)
                print(self.active_class)
         


    def execute(self):
        
        while True:
            cv2.imshow("aRGBi SLIC", cv2.resize(mark_boundaries(self.aRGBi, self.segments), (900,900)))
            cv2.imshow("aRGBi large", cv2.resize(self.aRGBi, (600,600)))
            cv2.setMouseCallback("aRGBi SLIC", self.click_event, 900 / map_step)
            cv2.setMouseCallback("aRGBi large", self.click_event, 600 / map_step)
            k = cv2.waitKey(1)#아스키 코드로 키보드 입력 반환
           
            if k == 27: #ESC 
                cv2.destroyAllWindows()
                break
            if k == 100: # d diseased 100  d 키
                self.active_class = 0
            if k == 104: # h leaf   104    h 키
                self.active_class = 1
            if k == 98: # b background     b 키
                self.active_class = 2
            if k == 114: # r      114      r 키 draw 모드 (선택 가능 모드)
                self.mode = "draw"
                
            if k == 111: # o     111       o z키 observe 모드
                self.mode = "observe"
            if k == 99: #c    99           c 키 나머지 부분 색상 처리
                self.colorizeRest()
            if k == 115: # s 저장  115
                # ssdir = f'{self.patch_save_dir}\\SEGMENTS\\segment_{self.block_num}_'
                # cv2.imwrite(ssdir + "img_ori.jpg", self.aRGBi_m)
                self._saveBtn()
                print('ok')
            
                

source =  "D:\\gys_22_10_25_rgb\\" #"F:\\" "E:\\gys_22_10_28_rgb\\"
folder = "MOSA\\" # "MOSAIC\\""MOSA\\
ahsi_filename = "gys_22_10_25" # "gys_22_10_28_1" "gys_22_10_25"
extension = ".tif"
# ahsi = spec.open_image(source + folder + ahsi_filename + extension)
# ahsi = cv2.imread()h


vipshome = "D:\\vips-dev-8.14\\bin"
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

import pyvips

path = source + folder + ahsi_filename + extension

image = pyvips.Image.new_from_file(path, access='sequential')
print(image.width)
print(image.height)

init_folder = False
block_num = 0
for y in range(0, image.height - map_step, map_step):
    for x in range(0, image.width - map_step, map_step):
        if block_num >316:# 146/200/ 173/ 226 / 255  / 310  /450  172ee
            segmentor = Segmentor(source = source,
                            ahsi = image, 
                            save_dir = ahsi_filename,
                            y = y,
                            x = x, 
                            block_num = block_num, init_folder = init_folder)
            segmentor.execute()
        block_num += 1



from PIL import Image

class Img(Dataset):
    def __init__(self, img_df, _3d=True, transform=None):
        self.img_df = img_df
        self.transform = transform
        self._3d = _3d

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        img_path = self.img_df['path'].iloc[idx]
        
        # Use PIL to open the image file.
        img = Image.open(img_path)
        
        # Convert the image to a NumPy array.
        img = np.array(img)
        
        label = self.img_df['class'].iloc[idx]
        
        if self._3d:
            image = torch.zeros((1, 3, img.shape[0], img.shape[1]), dtype=torch.float)
            for i in range(3):   # RGB images have only three channels.
                if self.transform:
                    image[:,i,:,:] = torch.from_numpy(self.transform(img[:,:,i]))
                else:
                    image[:,i,:,:] = torch.from_numpy(img[:,:,i])
        
        else:
            image = torch.zeros((3, img.shape[0], img.shape[1]), dtype=torch.float)
            for i in range(3):   # RGB images have only three channels.
                if self.transform:
                    image[i,:,:] = torch.from_numpy(self.transform(img[:,:,i]))
                else:
                    image[i,:,:] = torch.from_numpy(img[:,:,i])
                
        
        return image.float(), label