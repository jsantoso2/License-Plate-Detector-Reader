# License-Plate-Detector-Reader
#### Detects and Reads Indonesian License Plate from Input Image/Video

### Examples:
<p align="center"> <img src=https://github.com/jsantoso2/License-Plate-Detector-Reader/blob/master/Screenshots/demo-1.JPG width="700"></p>
<p align="center">Demo for Images<p align="center">

<p align="center"> <img src=https://github.com/jsantoso2/License-Plate-Detector-Reader/blob/master/Screenshots/demo2.gif></p>
<p align="center">Demo for Videos<p align="center">
<p align="center">Source Video: https://www.youtube.com/watch?v=nZTsFLFZYR8 <p align="center">


**WARNING! Video will take significant computation times, Please use GPU to speed up computations!**

### Dataset
- ~1200 manually labelled dataset (from Google Images, https://garasi.id/, https://www.carmudi.co.id/)
- ~500 from Kaggle dataset (https://www.kaggle.com/imamdigmi/indonesian-plate-number)
- Total: 1472 Train Images, 201 Validation Images

### Tools/Framework Used
- To manually label images: LabelImg (http://tzutalin.github.io/labelImg/)
- ML models: YOLOv3 (https://pjreddie.com/darknet/yolo/), Keras-OCR (https://github.com/faustomorales/keras-ocr)
- Evaluation of ML models: mAP tool (https://github.com/Cartucho/mAP)
- Web Application: Streamlit (https://www.streamlit.io/)
- Deployment: Google Cloud Platform App Engine
- Data Cleaning/Manipulation: Python 

### Procedure
- Manually label dataset using LabelImg (http://tzutalin.github.io/labelImg/)
- Convert labels into YOLOv3 format
- Fine Tune YOLOv3 Model (https://pjreddie.com/darknet/yolo/)
- Predict on images/videos
- Pass prediction to Text Recognizer Keras-OCR (https://github.com/faustomorales/keras-ocr)
- Output final images/videos

### Results
- For YOLOv3 Fine-Tuning
<p align="center"> <img src=https://github.com/jsantoso2/License-Plate-Detector/blob/master/Screenshots/results1-1.JPG height="450"></p>
<p align="center">Training Loss, Validation mAP over Iterations <p align="center">
<p align="center"> <img src=https://github.com/jsantoso2/License-Plate-Detector/blob/master/Screenshots/results1-2.JPG height="250"></p>
<p align="center">Precision Recall Curve, True and False Positives<p align="center">
  
- For Keras-OCR (Text Recognizer)
  - Keras-OCR model was not fine-tuned. Data table below directly taken from https://github.com/faustomorales/keras-ocr
  - Keras-OCR (scale_2) was used for this instance
    | model                 | latency | precision | recall |
    |-----------------------|---------|-----------|--------|
    | [AWS](https://www.mediafire.com/file/7obsgyzg7z1ltb0/aws_annotations.json/file)                   | 719ms   | 0.45      | 0.48   |
    | [GCP](https://www.mediafire.com/file/8is5pq161ui95ox/google_annotations.json/file)                   | 388ms   | 0.53      | 0.58   |
    | [keras-ocr](https://www.mediafire.com/file/1gcwtrzy537v0sn/keras_ocr_annotations_scale_2.json/file) (scale=2)  | 417ms   | 0.53      | 0.54   |
    | [keras-ocr](https://www.mediafire.com/file/dc7e66oupelsp7p/keras_ocr_annotations_scale_3.json/file) (scale=3)  | 699ms   | 0.5       | 0.59   |

### Streamlit Web Application Screenshots
<p align="center"> <img src=https://github.com/jsantoso2/License-Plate-Detector-Reader/blob/master/Screenshots/app-1.JPG height="600"></p>
<p align="center">App Interface 1<p align="center">
<p align="center"> <img src=https://github.com/jsantoso2/License-Plate-Detector-Reader/blob/master/Screenshots/app-2.JPG height="600"></p>
<p align="center">App Interface 2<p align="center">
<p align="center"> <img src=https://github.com/jsantoso2/License-Plate-Detector-Reader/blob/master/Screenshots/app-4.JPG height="600"></p>
<p align="center">App Interface 3<p align="center">

### To Deploy Application to Google Cloud Engine:
- Deploy ready-application located in docker_test folder
- To deploy application to Google Cloud Engine:
  - Ensure that gcloud sdk is installed in local file system (https://cloud.google.com/sdk/install)
  - To list of all projects: gcloud projects list
  - To look at current project: gcloud config get-value project
  - Change to desired project: gcloud config set project (projectID)
  - To Deploy: gcloud app deploy

### References/Inspirations:
- https://www.youtube.com/watch?v=CnYfsu2SxZw
- https://towardsdatascience.com/automatic-license-plate-detection-recognition-using-deep-learning-624def07eaaf
- https://towardsdatascience.com/i-built-a-diy-license-plate-reader-with-a-raspberry-pi-and-machine-learning-7e428d3c7401

### Final Notes:
- To see how application works, please see Instructions.mp4 video
- To see more technical details, please see notes.docs for all my detailed notes
