from django.shortcuts import get_object_or_404, render, redirect
import cv2
import shutil
import numpy as np
import logging
from .models import Records
from sklearn.model_selection import train_test_split
from . import dataset_fetch as df
from . import cascade as casc
from PIL import Image
from django.contrib.auth.models import auth,User
from django.contrib.auth import authenticate, login
from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
from django.contrib.auth.decorators import login_required
from .models import Records
from aiproject.settings import BASE_DIR
from twilio.rest import Client

# Create your views here.
@login_required(login_url='login')
def index(request):
    return render(request,'index.html')

@login_required(login_url='login')
def errorImg(request):
    return render(request,'error.html')

def adminpage(request):
    return render(request,'adminpage.html')

def viewdetail(request):
    record_details = Records.objects.order_by('id')
    return render(request,'viewdetail.html',context={'record_details': record_details})

def posts_edit(request, id=None):
    instance = Records.objects.get(id=id)
    context={
        'instance': instance
    }
    return render(request, 'modal.html', context)

def deleterow(request,id):
    emp = Records.objects.get(id=id)
    emp.delete()
    return redirect('viewdetail')


def editimage(request,id):
    emp = Records.objects.get(id=id)
    # print(id)
    emp.file_image = request.FILES['imgfile']
    emp.save()
    return redirect('viewdetail')


@login_required(login_url='login')
def add_record(request):
    if request.method == "POST":
        fname = request.POST['fname']
        lname = request.POST['lname']
        id = request.POST['id']
        education1 = request.POST['edu']
        coun = request.POST['country']
        residence = request.POST['res']
        occupation = request.POST['occupation']
        m = request.POST['ms']
        bio = request.POST['bio']
        fin = None
        if request.FILES != 0:
            fin = request.FILES['img1']

        p = Records(id=id,first_name=fname,last_name=lname,residence=residence,country=coun,file_image=fin,education=education1,occupation=occupation,marital_status=m,bio=bio)
        p.save()
        return render(request,'Add_record.html')
    else:
        return render(request,'Add_record.html')

def updaterow(request):
    if request.method == 'POST':
        id = request.POST['id']
        r = Records.objects.get(id=id)
        r.first_name = request.POST['fname']
        r.last_name = request.POST['lname']
        r.country = request.POST['country']
        r.residence = request.POST['residence']
        r.education = request.POST['edu']
        r.occupation = request.POST['occ']
        r.marital_status = request.POST['ms']
        r.bio = request.POST['pwd']
        r.save()
        return redirect('viewdetail')
    else:
        return redirect('viewdetail')

def adminlogin(request):
    if request.method == 'POST':
        username1 = request.POST.get('username')
        password1 = request.POST.get('password')

        user = authenticate(username=username1,password=password1)

        if user is not None:
            if User.objects.filter(username=username1,is_superuser=True):
                return render(request,'adminpage.html')
            else:
                return redirect('index')
        else:
            return render(request,'admin_login.html')
    else:
        return render(request,'admin_login.html')





def login_view(request):
    if request.method == 'POST':
        username1 = request.POST.get('username')
        password1 = request.POST.get('password')

        user = authenticate(username=username1,password=password1)

        if user is not None:
            # if User.objects.filter(username=username1,is_superuser=True):
            #     return redirect('admin_page')
            # else:
            login(request, user)
            return redirect('index')
        else:
            return redirect('/')
    else:
        return render(request,'login.html')


def registration(request):
    if request.method == 'POST':
        
        username1 = request.POST['username']
        password1 = request.POST['password']
        email = request.POST['email']
        user1 = User.objects.create_user(username=username1,password=password1,email=email)
        user1.save()
        return redirect('/')
    else:    
        return render(request,'registration.html')

@login_required(login_url='login')
def detect_image(request):
    userImage = request.FILES['userImage']

    svm_pkl_filename =  str(BASE_DIR) + '/ml/serializer/svm_classifier.pkl'

    svm_model_pkl = open(svm_pkl_filename, 'rb')
    svm_model = pickle.load(svm_model_pkl)
    #print "Loaded SVM model :: ", svm_model

    pca_pkl_filename =  str(BASE_DIR) + '/ml/serializer/pca_state.pkl'

    pca_model_pkl = open(pca_pkl_filename, 'rb')
    pca = pickle.load(pca_model_pkl)
    #print pca

    '''
    First Save image as cv2.imread only accepts path
    '''
    im = Image.open(userImage)
    #im.show()
    imgPath = str(BASE_DIR)+'/ml/uploadedImages/'+str(userImage)
    im.save(imgPath, 'JPEG')

    '''
    Input Image
    '''
    try:
        inputImg = casc.facecrop(imgPath)
        inputImg.show()
    except :
        print("No face detected, or image not recognized")
        return redirect('error_image')

    imgNp = np.array(inputImg, 'uint8')
    #Converting 2D array into 1D
    imgFlatten = imgNp.flatten()
    #print imgFlatten
    #print imgNp
    imgArrTwoD = []
    imgArrTwoD.append(imgFlatten)
    # Applyting pca
    img_pca = pca.transform(imgArrTwoD)
    #print img_pca

    pred = svm_model.predict(img_pca)
    print(svm_model.best_estimator_)
    print(pred[0])

    return redirect('index')

@login_required(login_url='login')
def create_dataset(request):
    #print request.POST
    userId = request.POST['userId']
    # print(cv2.__version__)
    # Detect face
    # Creating a cascade image classifier
    faceDetect = cv2.CascadeClassifier(str(BASE_DIR)+'/ml/haarcascade_frontalface_default.xml')
    # capture images from the webcam and process and detect the face
    # takes video capture id, for webcam most of the time its 0.
    cam = cv2.VideoCapture(0)

    # Our identifier
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is
    id = userId
    originalImage = False
    # Our dataset naming counter
    sampleNum = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while(sampleNum < 30):
        # Capturing the image
        #cam.read will return the status variable and the captured colored image
        ret, img = cam.read()
        #the returned img is a colored image but for the classifier to work we need a greyscale image
        #to convert
        if(not originalImage):
            cv2.imwrite(str(BASE_DIR)+'/static/img/'+str(id)+'.jpg',img)
            originalImage = True
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #To store the faces
        #This will detect all the images in the current frame, and it will return the coordinates of the faces
        #Takes in image and some other parameter for accurate result
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        #In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
        for(x,y,w,h) in faces:
            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum+1
            # Saving the image dataset, but only the face part, cropping the rest
            cv2.imwrite(str(BASE_DIR)+'/ml/dataset/user.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(250)

        #Showing the image in another window
        #Creates a window with window name "Face" and with the image img
        cv2.imshow("Face",img)
        #Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        #To get out of the loop
        if(sampleNum>35):
            break
    #releasing the cam
    cam.release()
    # destroying all the windows
    cv2.destroyAllWindows()
    record = Records()
    record.id = id 
    record.first_name = "N/A"
    #record.save()

    return redirect('index')

@login_required(login_url='login')
def detect(request):
    ACCOUNT_SID = 'ACcbce6eb291169a5eff298a633a4cba81'
    AUTH_TOKEN = 'd9724d763acda393f33c3cf0186ccf61'
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    faceDetect = cv2.CascadeClassifier(str(BASE_DIR)+'/ml/haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    # creating recognizer
    rec =cv2.face.LBPHFaceRecognizer_create();
    # loading the training data
    rec.read(str(BASE_DIR)+'/ml/recognizer/trainingData.yml')
    getId = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    userId = 0
    count_i=0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)

            getId,conf = rec.predict(gray[y:y+h, x:x+w]) #This will predict the id of the face

            if conf<35:
                userId = getId
                cv2.putText(img, "Detected",(x,y+h), font, 2, (0,255,0),2)
                print(count_i)
            elif count_i<50:
                cv2.putText(img, "Unknown",(x,y+h), font, 2, (0,0,255),2)
                count_i+=1
            elif count_i>=50:
                cv2.putText(img, "Unknown",(x,y+h), font, 2, (0,0,255),2)
                message = client.messages.create(to="+918217352361", from_="+12054481604",body="Unknown Person Found!!")
                cam.release()
                cv2.destroyAllWindows()
                return redirect('index')
            else:
                cv2.putText(img, "Unknown",(x,y+h), font, 2, (0,0,255),2)

            # Printing that number below the face
            # @Prams cam image, id, location,font style, color, stroke

        cv2.imshow("Face",img)
        if(cv2.waitKey(1) == ord('q')):
            break
        elif userId != 0:
            cv2.waitKey(1000)
            cam.release()
            cv2.destroyAllWindows()
            return redirect('webcam/details/'+str(userId))

    cam.release()
    cv2.destroyAllWindows()
    return redirect('index')

@login_required(login_url='login')
def details(request, id):
    try:
        record = Records.objects.get(id=id)
    except:
        ACCOUNT_SID = 'ACcbce6eb291169a5eff298a633a4cba81'
        AUTH_TOKEN = 'd9724d763acda393f33c3cf0186ccf61'
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(to="+918217352361", from_="+12054481604",body="Unknown Person Found!!")
        return render(request,'error.html')
    context = {
        'record' : record
    }
    return render(request, 'details.html', context)

@login_required(login_url='login')
def eigen_train(request):
    path = str(BASE_DIR)+'/ml/dataset'

    # Fetching training and testing dataset along with their image resolution(h,w)
    ids, faces, h, w= df.getImagesWithID(path)
    print('features'+str(faces.shape[1]))
    # Spliting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.25, random_state=42)
    #print ">>>>>>>>>>>>>>> "+str(y_test.size)
    n_classes = y_test.size
    target_names = ['Manjil Tamang', 'Marina Tamang','Anmol Chalise']
    n_components = 15
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()

    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)

    print("done in %0.3fs" % (time() - t0))
    eigenfaces = pca.components_.reshape((n_components, h, w))
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    # #############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("Predicted labels: ",y_pred)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    # print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    # #############################################################################
    # Qualitative evaluation of the predictions using matplotlib

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    # plot the gallery of the most significative eigenfaces
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    # plt.show()

    '''
        -- Saving classifier state with pickle
    '''
    svm_pkl_filename = BASE_DIR+'/ml/serializer/svm_classifier.pkl'
    # Open the file to save as pkl file
    svm_model_pkl = open(svm_pkl_filename, 'wb')
    pickle.dump(clf, svm_model_pkl)
    # Close the pickle instances
    svm_model_pkl.close()



    pca_pkl_filename = BASE_DIR+'/ml/serializer/pca_state.pkl'
    # Open the file to save as pkl file
    pca_pkl = open(pca_pkl_filename, 'wb')
    pickle.dump(pca, pca_pkl)
    # Close the pickle instances
    pca_pkl.close()

    plt.show()

    return redirect('index')

@login_required(login_url='login')
def trainer(request):
    '''
        In trainer.py we have to get all the samples from the dataset folder,
        for the trainer to recognize which id number is for which face.

        for that we need to extract all the relative path
        i.e. dataset/user.1.1.jpg, dataset/user.1.2.jpg, dataset/user.1.3.jpg
        for this python has a library called os
    '''
    import os
    from PIL import Image

    #Creating a recognizer to train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #Path of the samples
    path = str(BASE_DIR)+'/ml/dataset'
    list_of_dir = os.listdir(path)
    if len(list_of_dir)==0:
        return render(request,'error.html')

    # To get all the images, we need corresponing id
    def getImagesWithID(path):
        # create a list for the path for all the images that is available in the folder
        # from the path(dataset folder) this is listing all the directories and it is fetching the directories from each and every pictures
        # And putting them in 'f' and join method is appending the f(file name) to the path with the '/'
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #concatinate the path with the image name
        #print imagePaths
        # Now, we loop all the images and store that userid and the face with different image list
        faces = []
        Ids = []
        for imagePath in imagePaths:
            # First we have to open the image then we have to convert it into numpy array
            faceImg = Image.open(imagePath).convert('L') #convert it to grayscale
            # converting the PIL image to numpy array
            # @params takes image and convertion format
            faceNp = np.array(faceImg, 'uint8')
            # Now we need to get the user id, which we can get from the name of the picture
            # for this we have to slit the path() i.e dataset/user.1.7.jpg with path splitter and then get the second part only i.e. user.1.7.jpg
            # Then we split the second part with . splitter
            # Initially in string format so hance have to convert into int format
            ID = int(os.path.split(imagePath)[-1].split('.')[1]) # -1 so that it will count from backwards and slipt the second index of the '.' Hence id
            # Images
            faces.append(faceNp)
            # Label
            Ids.append(ID)
            #print ID
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        return np.array(Ids), np.array(faces)

    # Fetching ids and faces
    ids, faces = getImagesWithID(path)

    #Training the recognizer
    # For that we need face samples and corresponding labels
    recognizer.train(faces, ids)

    # Save the recogzier state so that we can access it later
    recognizer.save(str(BASE_DIR)+'/ml/recognizer/trainingData.yml')
    cv2.destroyAllWindows()

    return redirect('index')

