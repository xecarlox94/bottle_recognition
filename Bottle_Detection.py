#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO


# In[2]:


model = YOLO('yolov8n.pt')


# In[3]:


import mlflow


# In[18]:


from roboflow import Roboflow
rf = Roboflow(api_key="llgtjSRbLz3hdjjaUEfb")
project = rf.workspace("erkan-unal").project("detect-o9dby")
dataset = project.version(1).download("yolov8")


# In[4]:


dataset


# In[5]:


dataset_path = './detect-1'
dataset_yaml = dataset_path+'/detect-1/data.yaml'


# In[7]:


model.train(data=dataset_yaml, epochs=1, imgsz=640, device=0, batch=8)


# In[8]:


ultralytics-env


# In[14]:


x = torch.rand(5, 3)
print(x)


# In[9]:


torch.cuda.is_available()


# In[ ]:




