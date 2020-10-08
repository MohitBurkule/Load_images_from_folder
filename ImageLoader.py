class ImageLoader(object):
	try:
		import cv2
		import os
		import pickle
		import numpy as np
		print("Library Loaded Successfully ..........")
	except:
		print("Library not Found ! ")
    def __init__(self,PATH='', IMAGE_SIZE = 50):
        self.PATH = PATH
        self.IMAGE_SIZE = IMAGE_SIZE

        self.image_data = []
        self.x_data = []
        self.y_data = []
        self.CATEGORIES = []

        # This will get List of categories
        self.list_categories = []




if __name__ == "__main__":
    path = 'path_to_your_training_dataset_folder'
	#intended future usage 
    a = ImageLoader(PATH=path,DIRECTORY_STRUCTURE=1,MAX_IMAGES_PER_CAT=100,IMAGE_SIZE=80,)
	
	a.config_image(IMAGE_SIZE=80,FLATTERN=True,CUSTOM_PREPROCESS=function123)				
		
	a.save(SAVE_PATH='current_folder',PREFIX="train",SAVE_BATCH_SIZE=False) 
    
	X_Data,Y_Data = a.load(BATCH_NO=1,PREFIX="train") 

	
    print(X_Data.shape)

















