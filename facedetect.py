import cv2

trainedfacemodel = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)

while True:
    workingcorrectly, video = webcam.read()

    if not workingcorrectly:
        break

    blacknwhite = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)

    
    faces = trainedfacemodel.detectMultiScale(blacknwhite)

    
    for (x, y, w, h) in faces:
        cv2.rectangle(video, (x, y), (x + w, y + h), (0, 255, 0), 2)

  
    cv2.imshow("human", video)

    
    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
