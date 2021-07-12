# MelGAN-Tensorflow2

### Training

1. **한국어 음성 데이터 다운로드**

    * [KSS](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset)

2. **`~/MelGAN-Tensorflow2`에 학습 데이터 준비**

   ```
   MelGAN-Tensorflow2
     |- archive
         |- kss
             |- 1
             |- 2
             |- 3
             |- 4
         |- transcript.v.1.x.txt
   ```

3. **Preprocess**
   ```
   python preprocess.py
   ```
     * data 폴더에 학습에 필요한 파일들이 생성됩니다

4. **Train**
   ```
   python train.py
   ```

   재학습 시 가장 최근에 저장된 체크포인트 파일을 불러옵니다
   
5. **Inference**
   ```
   python inference.py
   ```
     * test 폴더에 wav 파일을 넣으면, 멜스펙트로그램으로 바꾼 후 melgan에 입력하고 output 폴더에 출력 wav가 생성됩니다



윈도우에서 MelGAN 학습하기
  * https://chldkato.tistory.com/144
  
MelGAN 정리
  * https://chldkato.tistory.com/142
