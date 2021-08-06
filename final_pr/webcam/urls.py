from django.urls import path

from . import views

urlpatterns = [
    # path('logindetails', views.logindetails, name='logindetails'),
    path('', views.login_view, name='login'),
    path('index', views.index, name='index'),
    path('registration', views.registration, name='registration'),
    path('errorImg',views.errorImg,name='ErrorImage'),
    path('detect_image',views.detect_image,name='detect_image'),
    path('create_dataset',views.create_dataset,name='create_dataset'),
    path('detect',views.detect,name='detect'),
    path('eigen_train',views.eigen_train,name='eigenTrain'),
    path('trainer', views.trainer, name='trainer'),
    path('webcam/details/<str:id>',views.details,name='details'),
    path('adminlogin', views.adminlogin, name='adminlogin'),
    path('adminpage', views.adminpage,name='adminpage'),
    path('viewdetail',views.viewdetail,name='viewdetail'),
    path('webcam/deleterow/<str:id>',views.deleterow,name='deleterow'),
    path('edit/<str:id>',views.posts_edit,name='edit'),
    path('add_record',views.add_record,name='add_record'),
    path('updaterow',views.updaterow,name='updaterow'),
    path('webcam/editimage/<str:id>',views.editimage,name='editimage'),
]