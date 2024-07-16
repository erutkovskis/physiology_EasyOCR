import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import re
import easyocr
import csv


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score
    
    scores = []
    angles = np.arange(-limit, limit+delta, delta)
    for angle in angles:
        histogram, score = determine_score(image, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)
    return best_angle, corrected

def preprocess_image(image):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve OCR accuracy
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #binarization/Otsu's thresholding
    _, binary = cv2.threshold(gray, 0 ,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #skew correction
    corrAngle, processed_image = correct_skew(binary)

    #thinning/skeletonization
    # not necessary because of the constant width

    return corrAngle, processed_image

def format_EtCO2(text):
    """
    Function to format EtCO2 value by inserting a decimal point if missing.
    Assumes the decimal point is missing between the last two digits.
    """
    if re.fullmatch(r'\d{2}', text):
        return text[0] + '.' + text[1]
    return text

def extract_text_from_video(video_path, isEtCO2, preprocess):
    cap = cv2.VideoCapture(video_path)
    detectedData = {}
    frameCnt = 0
    frameSkip = 15
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prepocess the frame
        if preprocess:
            corrAngle, processed_frame = preprocess_image(frame)
        else:
            processed_frame = frame
            corrAngle = 0

        data = reader.readtext(processed_frame, batch_size=8, allowlist='./:()0123456789', width_ths=0.5, contrast_ths=0.05, text_threshold=0.9)
        #optimBoxes = reader.detect(frame, optimal_num_chars = 2)
        
        for recognised in data:
            if isEtCO2:
                recognisedValue = format_EtCO2(recognised[1])
            else:
                recognisedValue = recognised[1]
            boxCoord = np.array(recognised[0],np.int32)
            boxCoord = boxCoord.reshape((-1,1,2))
            #labelRecognised = f'{recognised[1]}'
            labelRecognised = recognisedValue
            labelConfidence = f'{recognised[2]:.3f}'
            frame = cv2.polylines(frame,[boxCoord],True,(0,255,255)) # draw a box around the recognised character 
            frame = cv2.putText(frame,labelRecognised,(boxCoord[0][0][0],boxCoord[0][0][1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 3) # insert the recognised value
            frame = cv2.putText(frame,labelConfidence,(boxCoord[0][0][0]-25,boxCoord[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36, 255, 12), 1) # insert the confidence value

            timestamp = (int(cap.get(cv2.CAP_PROP_POS_MSEC))/1000.0)
            #detectedData({'Time (s)': timestamp,'Recognised': recognisedValue})
            detectedData[timestamp] = recognisedValue
            print(f'Array: {recognised}, Correction Angle: {corrAngle}, Recognised Value: {recognisedValue}')
            

        # for recogBox in optimBoxes[0]:
        #     for vertices in recogBox:
        #         optimBoxCoord = np.array(vertices,np.int32)
        #         frame = cv2.rectangle(frame,(optimBoxCoord[0],optimBoxCoord[2]),(optimBoxCoord[1],optimBoxCoord[3]),(0,255,255),2)



        cv2.imshow('Number Recognition', frame)
        cv2.imshow('Processed frame', processed_frame)
        frameCnt += frameSkip
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameCnt)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cap.release()
    cv2.destroyAllWindows
    print(detectedData)
    return(detectedData)

def write_to_csv(path_to_csv,detected_data):
    with open(path_to_csv, 'w', newline='') as csvfile:
        fieldnames = ['Time (s)'] + ['Detected']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # shape rows correctly
        row = {'Time (s)': None, 'Detected': None}
        for time in detected_data.keys():
            row['Time (s)'] = time
            row['Detected'] = detected_data[time]
            writer.writerow(row)


reader = easyocr.Reader(['en'])    
detected_data = extract_text_from_video('C:/Users/erutkovs/OneDrive - University College London/MRes sVNS project/Human trial/human_trial_recordings/data_20052024/video/008_20052024_C_card.mp4', 0, 0)
write_to_csv('C:/Users/erutkovs/OneDrive - University College London/MRes sVNS project/Human trial/human_trial_recordings/data_20052024/video/export_csv/008_20052024_C_card_easy_ocr.csv', detected_data)