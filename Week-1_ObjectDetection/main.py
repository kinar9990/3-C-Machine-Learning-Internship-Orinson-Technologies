from ultralytics import YOLO
import cv2
import os
import queue
import threading


class inference:
    def __init__(self, model_path, data , is_image = True):
        
        self.model_path = model_path # path to the model file
        self.data = data # path to the image file
        self.is_image = is_image # True if the data is an image, False if it's a video
        if not self.is_image: # if the data is a video
            self.imageQueue = queue.Queue(maxsize=1) # create a queue to store the images
        self.model = self.load_model() # load the model

    
    def load_model(self):
        model = YOLO(self.model_path) # load the model from the specified path
        model.info() # print the model information
        return model # return the model
    
    def load_image(self):
        # Loop through each file in the folder and add to a list
        images = []
        for filename in os.listdir(self.data): 
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')): # check if the file is an image
                file_path = os.path.join(self.data, filename) # construct the full file path ↓ 
                image = cv2.imread(file_path) # read the image
                if image is not None: # check if the image is read correctly
                    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert the image to RGB format for better detection
                    images.append(image_RGB)
        return images    #return the list of images
    
    def load_video(self):
        cap = cv2.VideoCapture(self.data) # load the video from the specified path
        while cap.isOpened():
            print(self.imageQueue.empty())
            ret, frame = cap.read() # read the video frame
            if ret and self.imageQueue.empty(): # check if the frame is read correctly and the queue is empty
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert the frame to RGB format for better detection
                self.imageQueue.put(frame)
            

    def predict(self, image):
        result = self.model.predict(image, # pass the image to the model
                                conf = 0.4, # confidence threshold
                                verbose = True # show detections stats in console
                                )
        detections = result[0].plot() # plot the detections
        return cv2.cvtColor(detections, cv2.COLOR_RGB2BGR) # convert the detections to BGR format
    
    def show(self, image):
        detections = self.predict(image) # predict the detections on each image
        cv2.namedWindow("Inference", cv2.WINDOW_NORMAL) # create a window to show the detections
        cv2.imshow("Inference", detections) # show the detections on the image
        
    
    def main(self):
        if self.is_image:
            images = self.load_image() # load the images
            print("Anykey to move forward")
            for image in images:
                self.show(image) # show the detections on each image
                cv2.waitKey(0) # wait for a key press
            cv2.destroyAllWindows() # destroy all the windows
        else:
            threading.Thread(target=self.load_video, daemon=True).start() # start a new thread to load the video
            print("q to quit")
            while True: # loop until the 'q' key is pressed
                image = self.imageQueue.get() # get the image from the queue
                self.show(image) # show the detections on the image
                if cv2.waitKey(1) & 0xFF == ord('q'): # check if the 'q' key is pressed
                    cv2.destroyAllWindows() # destroy all the windows
                    break # break the loop if the 'q' key is pressed


if __name__ == "__main__":
    model_path = "model/yolov10m.pt" # path to the model file
    data = "data" # path to the image file
    objectDetection = inference(model_path, data, is_image = True) # create an instance of the inference class
    objectDetection.main() # call the main method of the inference class
