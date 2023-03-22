from ultralytics import YOLO
import cv2
import supervision as sv
from collections import deque
import numpy as np
from PIL import Image,ImageFilter
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]
dic = {}
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_trail(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    #cv2.line(img, line[0], line[1], (46,162,112), 3)

    #height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(dic):
      if key not in identities:
        dic.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0
        # create new buffer for new object
        if id not in dic:  
          dic[id] = deque(maxlen= 64)
        color = compute_color_for_labels(object_id[i])
        #obj_name = names[object_id[i]]
        #label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

        # add center to buffer
        dic[id].appendleft(center)
        # draw trail
        for i in range(1, len(dic[id])):
            # check if on buffer value is none
            if dic[id][i - 1] is None or dic[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, dic[id][i - 1], dic[id][i], color, thickness)
    return img
def main():
    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 1,
        text_scale = 0.5
    )
    model = YOLO("yolov8l.pt")
    dic = dict()
    frame_cnt = 0
    for i,result in enumerate(model.track(source='01_002.avi',show=False,stream=True)):
        frame = result.orig_img
        org = np.copy(frame)
        detections = sv.Detections.from_yolov8(result)
        try:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        except:
            pass
        labels = [
            f"OBJECT-ID: {tracker_id} CLASS: {model.model.names[class_id]} CF: {confidence:0.2f}"
            for _,confidence,class_id,tracker_id in detections
        ]
        id = detections.tracker_id
        xyxy = detections.xyxy
        #frame = box_annotator.annotate(scene=frame,detections=detections,labels=labels)
        center_x = int((xyxy[0][0]+xyxy[0][2])/2)
        center_y = int((xyxy[0][1]+xyxy[0][3])/2)
        center = (center_x,center_y)
        identities = id
        img = np.zeros((480,640,3))
        cv2.line(img,(230,0),(230,640),(255,255,255),5)
    
        draw_trail(img, xyxy, model.model.names, id,identities)
        cv2.imshow("yolov8_f",img)
        if cv2.waitKey(30) == 27:
            break
        frame_cnt+=1
main()