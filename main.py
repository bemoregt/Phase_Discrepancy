import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageProcessingApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Subtraction using Fourier Phase")
        master.geometry("1600x800")

        # 이미지 로드 버튼
        self.load_button = tk.Button(master, text="이미지 로드", command=self.load_images)
        self.load_button.pack(pady=10)

        # 캔버스 프레임 생성
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack(expand=True, fill=tk.BOTH)

        # 원본 이미지와 결과 이미지를 표시할 레이블 생성
        self.img1_label = tk.Label(self.canvas_frame)
        self.img1_label.grid(row=0, column=0, padx=5, pady=10)

        self.img2_label = tk.Label(self.canvas_frame)
        self.img2_label.grid(row=0, column=1, padx=5, pady=10)

        self.result_label = tk.Label(self.canvas_frame)
        self.result_label.grid(row=0, column=2, padx=5, pady=10)

        # 이미지 처리 버튼
        self.process_button = tk.Button(master, text="Phase Discrepancy", command=self.process_images, state=tk.DISABLED)
        self.process_button.pack(pady=10)

    def load_images(self):
        # 이미지 파일 dialog로 선택
        filename = filedialog.askopenfilename(title="Image Load")
        if filename:
            # 이미지 로드 및 좌우 분할
            img = cv2.imread(filename)
            height, width = img.shape[:2]
            
            # 이미지를 좌우 2등분
            self.img1 = img[:, :width//2]
            self.img2 = img[:, width//2:]

            # OpenCV BGR to RGB 변환
            img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)

            # tkinter에 이미지 표시
            self.display_image(self.img1_label, img1_rgb)
            self.display_image(self.img2_label, img2_rgb)

            # 처리 버튼 활성화
            self.process_button.config(state=tk.NORMAL)

    def display_image(self, label, image, cmap=None):
        # OpenCV 이미지를 tkinter 레이블에 표시
        from PIL import Image, ImageTk
        import matplotlib.pyplot as plt
        import io
        
        if cmap:
            # matplotlib를 사용해 컬러맵 적용
            fig, ax = plt.subplots(figsize=(5,5))
            ax.imshow(image, cmap=cmap)
            ax.axis('off')
            plt.tight_layout()
            
            # 메모리 버퍼에 이미지 저장
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            # 버퍼에서 이미지 로드
            buf.seek(0)
            img = Image.open(buf)
            img = img.resize((400, 400), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
        else:
            # 일반 그레이스케일 이미지 처리
            img = Image.fromarray(image)
            img = img.resize((400, 400), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
        
        label.config(image=photo)
        label.image = photo

    def process_images(self):
        # 그레이스케일로 변환
        img1_gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        # 푸리에 변환
        d1 = np.fft.fft2(img1_gray)
        d2 = np.fft.fft2(img2_gray)

        # 위상과 진폭 추출
        phase1 = np.angle(d1)
        amp1 = np.abs(d1)
        phase2 = np.angle(d2)
        amp2 = np.abs(d2)

        # 복소수 재구성
        def complex_numpy(real, imag):
            return real * np.exp(1j * imag)

        z1 = complex_numpy((amp1-amp2), phase1)
        z2 = complex_numpy((amp2-amp1), phase2)

        # 역 푸리에 변환
        m1 = np.fft.ifft2(z1)
        m2 = np.fft.ifft2(z2)

        # 결과 처리
        m11 = np.abs(m1)
        m22 = np.abs(m2)
        m12 = np.multiply(m11, m22)
        result = np.interp(m12, (m12.min(), m12.max()), (0, 255)).astype(np.uint8)

        # 결과 이미지 표시
        self.display_image(self.result_label, result, cmap='hsv')

def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()