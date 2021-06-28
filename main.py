from flask import Flask, render_template, request, redirect
import os
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from numpy.linalg import norm
from skfuzzy.membership import gaussmf


def brightness(img):
    if len(img.shape) == 3:
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        return np.average(img)

def adjust_brightness(img, thresh=100):
	# Checking Brigtness Level
	c=0
	if brightness(img)<thresh:
		# setting a value for adjusting brightness level
		value = int(thresh-brightness(img))

		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsv)

		lim = 255 - value
		v[v > lim] = 255
		v[v <= lim] += value

		final_hsv = cv2.merge((h, s, v))
		img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

		# Recurssion 
		if gaussmf(np.asarray([brightness(img)]),thresh,50) <0.95 and c<=100:
			c+=1; 
			adjust_brightness(img)

		return img

	else:
		value = int(brightness(img)-thresh)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h, s, v = cv2.split(hsv)

		lim = 255 - value
		v[v > lim] = 255
		v[v <= lim] -= value

		final_hsv = cv2.merge((h, s, v))
		img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

		if gaussmf(np.asarray([brightness(img)]),thresh,50) <0.95:
			c+=1
			adjust_brightness(img)

		return img


input_size = 50*50*3 # image shape
hidden_layers = 100
num_classes = 37


import torchvision
import torch
import torch.nn as nn

class net(nn.Module):

  def __init__(self,input_size,hidden_layers,num_classes):

    super(net,self).__init__()
    self.input = nn.Linear(in_features=50*50*3,out_features=1024)
    self.relu_1 = nn.ReLU()
    self.hidden1 = nn.Linear(in_features=1024,out_features=2048)
    self.relu_2 = nn.ReLU()
    self.hidden2 = nn.Linear(in_features=2048,out_features=2048)
    self.relu_3 = nn.ReLU()
    self.hidden3 = nn.Linear(in_features=2048,out_features=2048)
    self.relu_4 = nn.ReLU()
    self.hidden4 = nn.Linear(in_features=2048,out_features=1024)
    self.relu_5 = nn.ReLU()
    self.hidden5 = nn.Linear(in_features=1024,out_features=512)
    self.relu_6 = nn.ReLU()
    self.output = nn.Linear(in_features=512,out_features=num_classes)
    

  def forward(self,X):
    model = self.input(X)
    model = self.relu_1(model)
    model = self.hidden1(model)
    model = self.relu_2(model)
    model = self.hidden2(model)
    model = self.relu_3(model)
    model = self.hidden3(model)
    model = self.relu_4(model)
    model = self.hidden4(model)
    model = self.relu_5(model)
    model = self.hidden5(model)
    model = self.relu_6(model)
    model = self.output(model)

    return model

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/data',methods = ['GET','POST'])
def data():
    if request.method == 'POST':
        if request.files:

            try:
                os.mkdir('data/')
            except:
                pass
            image = request.files['image']
            image.save('data/'+'test'+'.png')

            im = cv2.imread('data/test.png')
            im = cv2.resize(im, (50,50))
            im = adjust_brightness(im)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, bw = cv2.threshold(gray,100,256,cv2.THRESH_BINARY_INV)
            cv2.imwrite('image_binirised.jpg',(bw))
            img = cv2.imread('image_binirised.jpg')


            model = net(input_size,hidden_layers,num_classes)
            model.load_state_dict(torch.load('model.pth',map_location=torch.device('cpu')))
            img = torchvision.transforms.functional.to_tensor(img).reshape(-1,50*50*3)
            print(torch.topk(model(img.to('cpu'))).detach().numpy())
            messege = "The predicted sign is "+str(torch.topk(model(img.to('cpu')), 1)[1][0][0].detach().numpy())


            
            return render_template('index.html',messege = messege)



if __name__ == '__main__':
    app.run(debug = True)








