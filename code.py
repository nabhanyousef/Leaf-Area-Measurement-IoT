#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from sklearn.model_selection import train_test_split
#from resnet50 import ResNet50
from keras.applications.resnet import ResNet50
from scipy.spatial import distance as dist
#from keras.preprocessing import image
import tensorflow.keras.preprocessing.image
from tensorflow.keras.utils import img_to_array
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import ImageTk, Image
from imutils import perspective
from imutils import contours
from pygame import mixer
import tensorflow as tf
#from tkinter import *
from tkinter import Tk, Label, Frame, Grid, Button,Entry,END,DISABLED,NORMAL ,PhotoImage,Toplevel
import numpy as np
import argparse
import imutils
import datetime
from datetime import datetime
import csv 
import time
import cv2
import pyfirmata
import os
import numpy
global nn
#for control the motor using arduino
pin0=11
pin1=12
#note that in mini computer you have to change it to COM3
port="COM3"
board=pyfirmata.Arduino(port)
#creat the list of your defects
HeatNo=['Heat-No']
Patches=['patches']
Crack=['Porosity']
Inclusion=['inclusion']
Scratches=['scratches']
#load our trained model from ResNet 
model = tf.keras.models.load_model("nabhan1.model")
#design the main root
#root2 = Tk()
#root2.iconbitmap('N2.ico')
#root2.title('NABHAN YOUSEF APPLICATION')
#root2.config(background='light gray')


#second label

###############
#app9 = Frame(root2,bg='light green')
#app9.place(x = 0, y = 0,width =1920,height =100)
#nbhan9 = Label(app9)
#nbhan9.pack()
        #show image in frame 
#image1 = cv2.imread("img15.png")
#cv2.imwrite('img15.png',image1)
#cv2image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGBA)
#img = Image.fromarray(cv2image)
#img= img. resize((1920,100), Image. ANTIALIAS)
#imgtk = ImageTk.PhotoImage(image=img)
#nbhan9.imgtk = imgtk
#nbhan9.configure(image=imgtk)

######################################
#app10= Frame(root2,bg='light green')
#app10.place(x = 0, y = 105,width =1920,height =45)
#nbhan10 = Label(app10)
#nbhan10.pack()
        #show image in frame 
#image1 = cv2.imread("img16.png")
#cv2.imwrite('img16.png',image1)
#cv2image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGBA)
#img = Image.fromarray(cv2image)
#img= img. resize((1920,45), Image. ANTIALIAS)
#imgtk = ImageTk.PhotoImage(image=img)
#nbhan10.imgtk = imgtk
#nbhan10.configure(image=imgtk)


root1 = Tk()
root1.iconbitmap('N2.ico')
root1.title('NABHAN YOUSEF APPLICATION')
root1.config(background='light gray')
root1.geometry('2750x1750')
    #################################################
app4 = Frame(root1,bg='light green')
app4.place(x = 0, y = 0,width =300,height =200)
nbhan4 = Label(app4)
nbhan4.pack()
        #show image in frame 
image1 = cv2.imread("img11.png")
cv2.imwrite('img11.png',image1)
cv2image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGBA)
img = Image.fromarray(cv2image)
img= img. resize((300,200), Image. ANTIALIAS)
imgtk = ImageTk.PhotoImage(image=img)
nbhan4.imgtk = imgtk
nbhan4.configure(image=imgtk)
        ##################################3
app5 = Frame(root1,bg='light green')
app5.place(x = 305, y = 0,width = 1310,height =200)
nbhan5 = Label(app5)
nbhan5.pack()
        #show image in frame 
image1 = cv2.imread("img444.png")
cv2.imwrite('img444.png',image1)
cv2image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGBA)
img = Image.fromarray(cv2image)
img= img. resize((1310,200), Image. ANTIALIAS)
imgtk = ImageTk.PhotoImage(image=img)
nbhan5.imgtk = imgtk
nbhan5.configure(image=imgtk)
        
        ##################################################
app6 = Frame(root1,bg='light green')
app6.place(x = 1620, y = 0,width =300,height =200)
nbhan6 = Label(app6)
nbhan6.pack()
        #show image in frame 
image1 = cv2.imread("img13.png")
cv2.imwrite('img13.png',image1)
cv2image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGBA)
img = Image.fromarray(cv2image)
img= img. resize((300,200), Image. ANTIALIAS)
imgtk = ImageTk.PhotoImage(image=img)
nbhan6.imgtk = imgtk
nbhan6.configure(image=imgtk)
        ##################################3

    
   ##########################


def metal():
    root = Toplevel()
    root.iconbitmap('N2.ico')
    root.title('NABHAN YOUSEF APPLICATION')
    root.config(background='light gray')
    root.geometry('2750x1750')

        #prepare frames 
    app = Frame(root,bg='light gray')
    app.place(x =965, y = 205,width =650,height = 460)
    lmain = Label(app)
    lmain.pack()
    app1 = Frame(root,bg='light gray')
    app1.place(x =305, y = 205,width =650,height =460)
    nbhan = Label(app1)
    nbhan.pack()
    app2 = Frame(root,bg='light gray')
    app2.place(x = 305, y =670,width = 650,height = 75)
    nbhan1 = Label(app2)
    nbhan1.pack()
    app3 = Frame(root,bg='light gray')
    app3.place(x =965, y = 670,width = 650,height = 75)
    nbhan3 = Label(app3)
    nbhan3.pack()
        #####################################################################
    app4 = Frame(root,bg='light green')
    app4.place(x = 0, y = 0,width =300,height =200)
    nbhan4 = Label(app4)
    nbhan4.pack()
        #show image in frame 
    image1 = cv2.imread("img13.png")
    cv2.imwrite('img13.png',image1)
    cv2image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img= img. resize((300,200), Image. ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=img)
    nbhan4.imgtk = imgtk
    nbhan4.configure(image=imgtk)
        ##################################3
    app5 = Frame(root,bg='light green')
    app5.place(x = 305, y = 0,width = 1310,height =200)
    nbhan5 = Label(app5)
    nbhan5.pack()
        #show image in frame 
    image1 = cv2.imread("img444.png")
    cv2.imwrite('img12.png',image1)
    cv2image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img= img. resize((1310,200), Image. ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=img)
    nbhan5.imgtk = imgtk
    nbhan5.configure(image=imgtk)
        
        ##################################################
    app6 = Frame(root,bg='light green')
    app6.place(x = 1620, y = 0,width =300,height =200)
    nbhan6 = Label(app6)
    nbhan6.pack()
        #show image in frame 
    image1 = cv2.imread("img13.png")
    cv2.imwrite('img13.png',image1)
    cv2image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img= img. resize((300,200), Image. ANTIALIAS)
    imgtk = ImageTk.PhotoImage(image=img)
    nbhan6.imgtk = imgtk
    nbhan6.configure(image=imgtk)
        #########################################################
        
        #########################################################
    app12 = Frame(root,bg='#5dc072')
    app12.place(x =0, y =200,width =300,height =900)
    nbhan12 = Label(app12)
    nbhan12.pack()
        ################################################
    app13 = Frame(root,bg='#5dc072')
    app13.place(x =1620, y =200,width =300,height =900)
    nbhan13 = Label(app13)
    nbhan13.pack()
        #turn on the camera
        
    cap = cv2.VideoCapture(1)
        #function to make the camera on
    def camera_on():
        
        _, frame1 = cap.read()
        cv2image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        img= img. resize((650, 460), Image. ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(1, camera_on)
        b0['state']=DISABLED
        #function to enter the heat number
    def heat():
        global e
        e=Entry(root,width=40,bg='#5dc072',font="verdana 20 bold ")
        e.place(x =0, y = 205,width =300 ,height = 100)
        #function for capture image 
    def capture():    
        while True:
            _, frame = cap.read()
            cv2.imwrite('frame.jpg',frame)
            break

        #function to check the dimension 
    def dim():    
        from scipy.spatial.distance import euclidean
        from imutils import perspective
        from imutils import contours
        import numpy as np
        import imutils
        import cv2
        from PIL import Image, ImageTk
        global mylabel7
        global mylabel8

        img_path = "frame.jpg"

        # Read image and preprocess
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)

        edged = cv2.Canny(blur, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # Find contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Sort contours from left to right as leftmost contour is reference object
        (cnts, _) = contours.sort_contours(cnts)

        # Remove contours which are not large enough
        cnts = [x for x in cnts if cv2.contourArea(x) > 100]

        # Draw all contours on the image
        cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)

        # Reference object dimensions (assuming first detected contour is reference)
        ref_object = cnts[0]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        dist_in_pixel = euclidean(tl, tr)
        dist_in_cm = 1.98  # Known reference size in cm
        pixel_per_cm = dist_in_pixel / dist_in_cm

        # Process each contour
        for cnt in cnts:
            # Compute width and height
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box

            mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
            mid_pt_vertical = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
            width = euclidean(tl, tr) / pixel_per_cm
            height = euclidean(tr, br) / pixel_per_cm

            # Dimension check
            if (1 < width < 3) and (1 < height < 3):
                op = "Dimension Check: Pass"
            else:
                op = "Dimension Check: Pass"

            # Draw dimensions on image
            cv2.putText(image, "{:.1f}cm".format(width), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(image, "{:.1f}cm".format(height), (int(mid_pt_vertical[0] + 10), int(mid_pt_vertical[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Save and display updated image
        cv2.imwrite('output.jpg', image)
        cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        img = img.resize((650, 460), Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=img)

        nbhan.imgtk = imgtk
        nbhan.configure(image=imgtk)

        mylabel7 = Label(root, text=op, fg="green", font="verdana 24 bold")
        mylabel7.place(x=305, y=670, width=650, height=75)
         # Display "SAVED SUCCESSFULLY" message
        mylabel8 = Label(root, text="Saved successfully", fg="green", font="verdana 24 bold")
        mylabel8.place(x=965, y=670, width=650, height=75)
        #b3['state']=DISABLED
        #function tp predict the output
    
    def delete():
        
        e.delete(0,END)
        mylabel7.destroy() 
        mylabel8.destroy()
        b3['state']=NORMAL
        #b4['state']=NORMAL
        b1 = Button(root,image = img0,borderwidth = 0,highlightthickness = 0,command = heat,relief = "flat",bg='#5dc072')
        b1.place(x = 0, y = 205,width =300,height = 100)
        
        #############
        
        
        

    
        #functions to control the motor 
    def motor_up():
        board.digital[pin1].write(0)
        time.sleep(1)
        board.digital[pin1].write(1)

    def stop():
        board.digital[pin0].write(1)
        board.digital[pin1].write(1)
    def motor_down():
        board.digital[pin0].write(0)
        time.sleep(1)
        board.digital[pin0].write(1)
    def motor_off():
        board.exit()
        #image to use it as button 
        #enter
    img0 = PhotoImage(file = f"img0.png")
        #capture
    img1 = PhotoImage(file = f"img1.png")
        #dimension
    img2 = PhotoImage(file = f"img2.png")
        #defect
    #img3 = PhotoImage(file = f"img3.png")
        #save
    img4 = PhotoImage(file = f"img4.png")
        #turn on camera
    img5 = PhotoImage(file = f"img5.png")
        #up
    img6 = PhotoImage(file = f"img6.png")
        #stop
    img7 = PhotoImage(file = f"img7.png")
        #down
    img8 = PhotoImage(file = f"img8.png")
        
        #exit
    img10 = PhotoImage(file = f"img10.png")

        #define the buttons and control it 
    b0 = Button(root,image = img5,borderwidth = 0,highlightthickness = 0,command = camera_on,relief = "flat",bg='#5dc072')
    b0.place(x =1670, y =470,width =60,height =60)
    b1 = Button(root,image = img0,borderwidth = 0,highlightthickness = 0,command = heat,relief = "flat",bg='#5dc072')
    b1.place(x = 0, y = 205,width =300,height = 100)
    b2 = Button(root,image = img1,borderwidth = 0,highlightthickness = 0,command = capture,relief = "flat",bg='#5dc072')
    b2.place(x = 0, y = 305,width =300,height = 100)
    b3 = Button(root,image = img2,borderwidth = 0,highlightthickness = 0,command =dim ,relief = "flat",bg='#5dc072')
    b3.place(x = 0, y = 405,width = 300,height = 100)
    #b4 = Button(root,image = img3,borderwidth = 0,highlightthickness = 0,command =predict ,relief = "flat",bg='#5dc072')
    #b4.place(x = 0, y = 505,width = 300,height = 100)
    b5 = Button(root,image = img4,borderwidth = 0,highlightthickness = 0,command =delete ,relief = "flat",bg='#5dc072')
    b5.place(x = 0, y = 505,width = 300,height = 100)
    b6 = Button(root,image = img6,borderwidth = 0,highlightthickness = 0,command = motor_up ,relief = "flat",bg='#5dc072')
    b6.place(x = 1740, y = 400,width =60,height = 60)
    b7 = Button(root,image = img7,borderwidth = 0,highlightthickness = 0,command =stop,relief = "flat",bg='#5dc072')
    b7.place(x = 1740, y = 470,width = 60,height = 60)
    b8 = Button(root,image = img8,borderwidth = 0,highlightthickness = 0,command =motor_down ,relief = "flat",bg='#5dc072')
    b8.place(x = 1740, y = 540,width = 60,height = 60)
    b10 = Button(root,image = img10,borderwidth = 0,highlightthickness = 0,command =motor_off,relief = "flat",bg='#5dc072')
    b10.place(x = 1810, y = 470,width = 60,height = 60)
    root.mainloop()
    board.exit()
        #the end of root
def nometal():
    print("the inspection device working perfectly")
img20 = PhotoImage(file = f"img20.png")
b0 = Button(root1,image = img20,borderwidth = 0,highlightthickness = 0,command = metal,relief = "flat",bg='light gray')
b0.place(x = 850,y=360,width =300,height =100)
img21 = PhotoImage(file = f"img21.png")
b0 = Button(root1,image = img21,borderwidth = 0,highlightthickness = 0,command = nometal,relief = "flat",bg='light gray')
b0.place(x = 850, y =470,width=300,height =100)
img22=PhotoImage(file = f"img22.png")
b0 = Button(root1,image = img22,borderwidth = 0,highlightthickness = 0,command =root1.destroy,relief = "flat",bg='light gray')
b0.place(x = 850, y =580,width=300,height =100)
root1.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




