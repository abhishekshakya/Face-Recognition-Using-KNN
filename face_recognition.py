import cv2 
import numpy as np 
import os


##-------------------------------------------------KNN ALGO------------------------
def distance(x,point):
	d = np.sqrt(sum((x-point)**2))
	return d

def knn(X,Y,point,k=5):
	dist = []
	m = X.shape[0]

	for i in range(m):
		d = distance(X[i],point)
		dist.append((d,Y[i]))

	dist = sorted(dist,key= lambda d: d[0])
	dist = np.array(dist[:k])

	#adding voting part
	classes = np.unique(np.array(Y))
	# print(classes)

	votes = np.zeros(len(classes))

	for d in dist:#farther points will contribute less in voting part
		votes[int(d[1])]+= 1/(d[0])

	# print(votes)
	pred = np.argmax(votes)

	return pred



#-----------------------------------------------------------------------------------------------




#camera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
labels = []
dataset_path = './data/'

class_id = 0
names = {}

#data preparation
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		target = class_id*np.ones((data_item.shape[0]))
		class_id += 1
		labels.append(target)

X = np.concatenate(face_data,axis=0)
Y = np.concatenate(labels,axis=0)

# print(face_dataset.shape)
# print(face_labels.shape)

#testing 

while True:
	ret,frame = cap.read()
	if ret==False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	for face in faces:
		x,y,w,h = face 

		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		out = knn(X,Y,face_section.flatten())

		#name and rectabge
		pred_name = names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)

	cv2.imshow("Face",frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
