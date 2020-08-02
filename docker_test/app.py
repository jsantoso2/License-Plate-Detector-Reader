# common imports
import numpy as np
import os
import streamlit as st
from PIL import Image
import time
import cv2
from gpuinfo import GPUInfo
from io import BytesIO
from ffmpy import FFmpeg
import tensorflow as tf

import keras_ocr
# from google.cloud import storage

#@st.cache
def load_model():
    # exceed quota so need to load data from GCS. Will download data to local from GCS
    # bucket_name = 'license_plate_detector_jonathan'
    # file_id = ['yolov3_custom_train_1800.weights', 'crnn_kurapan.h5', 'craft_mlt_25k.h5']

    # storage_client = storage.Client.from_service_account_json('auth_key.json')
    # bucket = storage_client.bucket(bucket_name)

    # for source_blob_name in file_id:
        # if source_blob_name not in os.listdir():
            # blob = bucket.blob(source_blob_name)
            # blob.download_to_filename(source_blob_name)
    
    # YOLO CONFIGURATIONS
    net = cv2.dnn.readNetFromDarknet('yolov3_custom_train.cfg', 'yolov3_custom_train_1800.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    
    # check if GPU is available else use CPU
    gpus = GPUInfo.check_empty()
    print('gpus: ', gpus)
    if gpus is not None:
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    else:
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    # keras-ocr will load from local file
    detector = keras_ocr.detection.Detector(weights='clovaai_general', 
                                            load_from_torch=False, 
                                            optimizer='adam', 
                                            backbone_name='vgg',
                                                    weights_path_local = 'craft_mlt_25k.h5')
    recognizer = keras_ocr.recognition.Recognizer(alphabet=None, 
                                                  weights='kurapan', 
                                                  build_params=None, 
                                                  weights_path = 'crnn_kurapan.h5')

    pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer)
    
    # return YOLOv3 model, KerasOCR model
    return [net, pipeline]


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom, pipeline):
    ### YOLO CONFIGURATIONS
    classes = ['LP']   #This is the class labels
    
    # get cropped LP for text preprocessing
    crop_img = frame[top:bottom, left:right]

    # invert image  (CANNOT convert to grayscale because the KERAS OCR expects 3 channels)
    crop_img = cv2.bitwise_not(crop_img)

    # expand image size for better recognition
    (h, w) = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (w*2, h*2), interpolation = cv2.INTER_AREA)

    start = time.time()
    prediction_groups = pipeline.recognize([crop_img], detection_kwargs={'detection_threshold': 0.5, 'text_threshold': 0.4, 'link_threshold': 0.4}, recognition_kwargs={})
    end = time.time()
    print('Text Processing time: ', (end - start) * 100, ' ms')


    # only take top 3 to prevent from reading bottom line which is the expiry date
    if len(prediction_groups[0]) >= 3:
        predictions = prediction_groups[0][0:3]
    else:
        predictions = prediction_groups[0]

    # extract left starting coordinates and rearrange to read from left to right
    left_start_coor = [elem[1][0][0] for elem in predictions]
    sort_order = np.argsort(left_start_coor)
    predictions = [x for _,x in sorted(zip(sort_order,predictions))]

    # get only the text
    prediction_text = [x[0].upper() for x in predictions]


    ## IF LEN == 3 THEN WE CAN PERFORM VALIDATIONS to HELP...
    if len(prediction_text) == 3:
        # Ensure that first character is originating from license plate
        all_regions = ['BL', 'BB', 'BK', 'BA', 'BM', 'BP', 'BG', 'BN', 'BE', 'BD', 'BH', 'B', 'D', 'F', 'T', 'E',
                       'Z', 'H', 'G', 'K', 'R', 'AA', 'AD', 'AB', 'L', 'W', 'N', 'P', 'AG', 'AE', 'S', 'M', 'DK',
                       'EA', 'DH', 'EB', 'ED', 'KB', 'DA', 'KH', 'KT', 'KU', 'DB', 'DL', 'DM', 'DN', 'DD', 'DC',
                       'DT', 'DE', 'DG', 'DS', 'PB']
      
        # check validation
        if prediction_text[0].upper() in all_regions:
            pass
        else:
            # if two characters check first and second characters
            curr_reading = prediction_text[0].upper()
            if len(curr_reading) == 2:
                if curr_reading[0] in all_regions:
                    prediction_text[0] = curr_reading[0]
                elif curr_reading[1] in all_regions:
                    prediction_text[0] = curr_reading[1]
                else:
                    pass
            else:
                pass
              

    # Draw a bounding box for License Plate
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    
    # Label license Plate
    sep = ' '
    final_text = sep.join(prediction_text)
    labelSize, baseLine = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top1 = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top1 - round(1.5*labelSize[1]) - 5), (left + round(1.5*labelSize[0]), top1 + baseLine - 5), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, final_text, (left, top1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    #Display the label at the bottom of the bounding box (LicensePlate)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1]) + (bottom - top + 20)), (left + round(1.5*labelSize[0]), top + baseLine + (bottom - top + 20)), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (left, top + (bottom - top + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    
    # return text prediction
    return prediction_text


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, pipeline):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    ### YOLO CONFIGURATIONS
    confThreshold = 0.5  #Confidence threshold
    nmsThreshold = 0.4  #Non-maximum suppression threshold
    
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    text_pred = ''
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        text_pred = drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height, pipeline)
    
    return text_pred


def inference(net, pipeline, file_type = 'image'):
    ''' 
        net = YOLOV3 model
        pipeline = KerasOCR model
        file_uploaded = uploaded file (bytesio)
        file_type = 'image' / 'video'
    '''
    ### YOLO CONFIGURATIONS
    inpWidth = 416  #608     #Width of network's input image
    inpHeight = 416 #608     #Height of network's input image
    
    
    # file uploaded must be image
    if file_type == 'image':
        cap = cv2.VideoCapture('image.jpg')
        
    # file uploaded must be video
    else:
        cap = cv2.VideoCapture('video.mp4')
        #vid_writer = cv2.VideoWriter('pred.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        vid_writer = cv2.VideoWriter('pred.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        
    # Process Images and Video
    while cv2.waitKey(1) < 0:
        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            cv2.waitKey(1000)
            break
        
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        text_pred = postprocess(frame, outs, pipeline)
        
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        print(label)

        # Write the frame with the detection boxes
        if file_type == 'image':
            cv2.imwrite('pred.jpg', frame.astype(np.uint8))
            return text_pred
        else:
            vid_writer.write(frame.astype(np.uint8))
            
    return ''



#def get_download_link(file_type = 'image'):
#    if file_type == 'image':
#        return f'<a href="pred.jpg" download="pred.jpg">Download Output Image</a>'
#    else:
#        return f'<a href="pred_outputs.mp4" download="pred_outputs.mp4">Download Output Video</a>'


def main():
    st.title('🇮🇩 Indonesia License Plate Detector 👁')
    st.write("This Project is inspired by [Achraf Khazri's Automatic License Plate Detection & Recognition](https://towardsdatascience.com/automatic-license-plate-detection-recognition-using-deep-learning-624def07eaaf).")
    st.write("And also [Robert Chiriac's I Built a DIY License Plate Reader with Raspberry PI and ML](https://towardsdatascience.com/i-built-a-diy-license-plate-reader-with-a-raspberry-pi-and-machine-learning-7e428d3c7401).")
    
    # explanation
    st.subheader('How does it work?')
    st.write("1. Upload an image in either JPG format / Upload a video in MP4 format")
    st.write("2. Wait for Image to render") 
    st.write("3. Click the Make Prediction Button to run the model.")
    
    # image demo
    st.subheader('Image Examples:')
    im = Image.open('demo_image.jpg')
    st.image(im, use_column_width=True)
    
    # video demo
    st.subheader('Video Examples:')
    vid_file = open('demo_video.mp4', 'rb')
    vid_bytes = vid_file.read()
    st.video(vid_bytes)    
    st.warning('WARNING: Video will take a significant computation times, please use GPU to speed up computations!')
        
    # Inputs
    st.write('')
    st.subheader('Input File')

    # load model
    with st.spinner("Loading Model ..."):
        [net, pipeline] = load_model()
    
    # file uploader
    st.write("Upload a File Image (.jpg) or Video (.mp4)")
    uploaded_file = st.file_uploader("Choose a file to upload", type=["jpg", "JPG", "mp4"])
    
    if uploaded_file is not None:
        # NO way to retrieve filename so use file_type
        file_type = 'image'
        try:            
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            file_type = 'image'

        except:
            st.video(uploaded_file)
            st.write('Uploaded Video', use_column_width = True)
            file_type = 'video'

        # save image/video to local CPU
        if file_type == 'image':
            # convert image to cv2 format and write to local
            image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite('image.jpg', image)
            #st.write('after local')
            #st.image('image.jpg')
        else:
            # Write the mp4 video to local
            with open("video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            #st.write('after local')
            #st.video('video.mp4')
        
        st.subheader('Output:')

        # get inference on image and display if button is clicked
        if st.button("Make Prediction"):
        
            start_time = time.time()
            
            # Some number in the range 0-1 (probabilities)
            with st.spinner("Doing Prediction..."):
                text_pred = inference(net, pipeline, file_type)
            
            end_time = time.time()

            st.subheader('Predictions: ')
            
            if file_type == 'image':
                image = Image.open('pred.jpg')
                st.image(image, caption='Prediction Image', use_column_width=True)
                
                # write vehicle type
                vehicle_type = 'Unknown'
                propinsi = 'Unknown'
                if len(text_pred) == 3:
                    if int(text_pred[1]) >= 1 and int(text_pred[1]) <= 1999:
                        vehicle_type = 'Car'
                    elif int(text_pred[1]) >= 2000 and int(text_pred[1]) <= 6999:
                        vehicle_type = 'Motorcycle'
                    elif int(text_pred[1]) >= 7000 and int(text_pred[1]) <= 7999:
                        vehicle_type = 'Bus'
                    elif int(text_pred[1]) >= 8000 and int(text_pred[1]) <= 9999:
                        vehicle_type = 'Truck'
                    
                    
                    if (text_pred[0]) == 'BL':
                        propinsi = 'Aceh'
                    elif (text_pred[0]) == 'BB' or (text_pred[0]) == 'BK':
                        propinsi = 'Sumatera Utara'
                    elif (text_pred[0]) == 'BA':
                        propinsi = 'Sumatera Barat'
                    elif (text_pred[0]) == 'BM':
                        propinsi = 'Riau'
                    elif (text_pred[0]) == 'BP':
                        propinsi = 'Kepulauan Riau'
                    elif (text_pred[0]) == 'BG':
                        propinsi = 'Sumatera Selatan'
                    elif (text_pred[0]) == 'BN':
                        propinsi = 'Bangka Belitung'
                    elif (text_pred[0]) == 'BE':
                        propinsi = 'Lampung'
                    elif (text_pred[0]) == 'BD':
                        propinsi = 'Bengkulu'
                    elif (text_pred[0]) == 'BH':
                        propinsi = 'Jambi'
                    elif (text_pred[0]) == 'B':
                        propinsi = 'DKI Jakarta'
                    elif (text_pred[0]) == 'D' or (text_pred[0]) == 'F' or (text_pred[0]) == 'T' or (text_pred[0]) == 'E' or (text_pred[0]) == 'Z':
                        propinsi = 'Jawa Barat'
                    elif (text_pred[0]) == 'H' or (text_pred[0]) == 'G' or (text_pred[0]) == 'K' or (text_pred[0]) == 'R' or (text_pred[0]) == 'AA' or (text_pred[0]) == 'AD':
                        propinsi = 'Jawa Tengah'
                    elif (text_pred[0]) == 'AB':
                        propinsi = 'DIY Yogyakarta'    
                    elif (text_pred[0]) == 'L' or (text_pred[0]) == 'W' or (text_pred[0]) == 'N' or (text_pred[0]) == 'P' or (text_pred[0]) == 'AG' or (text_pred[0]) == 'AE' or (text_pred[0]) == 'S' or (text_pred[0]) == 'M':
                        propinsi = 'Jawa Timur'
                    elif (text_pred[0]) == 'DK':
                        propinsi = 'Bali'
                    elif (text_pred[0]) == 'DR' or (text_pred[0]) == 'EA':
                        propinsi = 'Nusa Tenggara Barat'
                    elif (text_pred[0]) == 'DH' or (text_pred[0]) == 'EB' or (text_pred[0]) == 'ED':
                        propinsi = 'Nusa Tenggara Timur'
                    elif (text_pred[0]) == 'KB':
                        propinsi = 'Kalimantan Barat'
                    elif (text_pred[0]) == 'DA':
                        propinsi = 'Kalimantan Selatan'
                    elif (text_pred[0]) == 'KH':
                        propinsi = 'Kalimantan Tengah'
                    elif (text_pred[0]) == 'KT':
                        propinsi = 'Kalimantan Timur'
                    elif (text_pred[0]) == 'KU':
                        propinsi = 'Kalimantan Utara'
                    elif (text_pred[0]) == 'DB' or (text_pred[0]) == 'DL':
                        propinsi = 'Sulawesi Utara'
                    elif (text_pred[0]) == 'DM':
                        propinsi = 'Gorontalo'
                    elif (text_pred[0]) == 'DN':
                        propinsi = 'Sulawesi Tengah'
                    elif (text_pred[0]) == 'DD':
                        propinsi = 'Sulawesi Selatan'
                    elif (text_pred[0]) == 'DC':
                        propinsi = 'Sulawesi Barat'
                    elif (text_pred[0]) == 'DT':
                        propinsi = 'Sulawesi Tenggara'
                    elif (text_pred[0]) == 'DE':
                        propinsi = 'Maluku'                        
                    elif (text_pred[0]) == 'DG':
                        propinsi = 'Maluku Utara'
                    elif (text_pred[0]) == 'DS':
                        propinsi = 'Papua'
                    elif (text_pred[0]) == 'PB':
                        propinsi = 'Papua Barat'                            

                sep = ' '
                final_text = sep.join(text_pred)
                
                # download link
                #st.markdown(get_download_link('image'), unsafe_allow_html=True)
                
                # write license plate
                st.write('License Plate: ', final_text)
                st.write('Province: ', propinsi)
                st.write('Vehicle Type: ', vehicle_type)
                
                # write prediction time
                pred_time = end_time - start_time
                st.write('Prediction Time: ' + ' {0:.2f}'.format(pred_time) + ' seconds')
                
            else:
                # open video file
                # change codecs to h264
                ff = FFmpeg(inputs={'pred.mp4': None}, outputs={'pred_outputs.mp4': '-c:v h264'})
                ff.run()
                
                video_file = open('pred_outputs.mp4', 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                st.write('Predicted Video')
                
                # download link
                #st.markdown(get_download_link('video'), unsafe_allow_html=True)
            
                # write prediction time
                pred_time = end_time - start_time
                st.write('Prediction Time: ' + ' {0:.2f}'.format(pred_time) + ' seconds')
                             

    st.write('')
    st.subheader("What is under the hood?")
    st.write("YOLOv3 Model for Detection, Keras OCR for Text recognition, and Streamlit web application")
    # st.image(Image.open('logo.jpg'), use_column_width = True)


if __name__ == '__main__':
    main()