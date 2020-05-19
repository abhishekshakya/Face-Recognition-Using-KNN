# Read and show video stream, capture images
# Detect Faces and show bounding box (haarcascade)
# Flatten the largest face image (grayscale) and save in a numpy
# Repeat the above for multiple people to generate training data

import cv2
import numpy as np 

#camera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []

dataset_path = './data/'

file_name = input('Enter name\n')

while True:
	ret,frame = cap.read()

	if(ret==False):
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	faces = sorted(faces,key=lambda f: f[2]*f[3],reverse=True)

	for face in faces:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)

		#Crop out required face
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip += 1
		if skip%10 == 0:
			face_data.append(face_section)
			print(len(face_data))

		cv2.imshow("video",frame)
		cv2.imshow("face Section",face_section)



	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break


#convert face list into a numpy array
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save data
np.save(dataset_path+file_name+'.npy',face_data)

cap.release()
cv2.destroyAllWindows()