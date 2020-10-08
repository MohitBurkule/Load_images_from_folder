class ImageLoader(object):
    try:
      import cv2
      import os
      import pickle
      import numpy as np
      #print("Library Loaded Successfully ..........")
    except Exception as e :
      print("Could not load few libraries :",str(e))
      
    def log(self,msg):
      if(self.verbose==1):
        print(msg)  
    
    def __init__(self,PATH='',DIRECTORY_STRUCTURE=None,MAX_IMAGES_PER_CAT=None,IMAGE_SIZE=50,verbose=1):
        self.PATH = PATH
        self.IMAGE_SIZE = IMAGE_SIZE
        self.DIRECTORY_STRUCTURE=DIRECTORY_STRUCTURE
        self.MAX_IMAGES_PER_CAT=MAX_IMAGES_PER_CAT
        self.verbose=verbose

        self.image_data = []
        self.x_data = []
        self.y_data = []
        self.CATEGORIES = []

		# This will get List of categories
        self.list_categories = []

    def config_image(IMAGE_SIZE=80,FLATTERN=True,CUSTOM_PREPROCESS=function123):
      pass
    def save(SAVE_PATH='current_folder',PREFIX="train",SAVE_BATCH_SIZE=False):
      pass 
    def load(BATCH_NO=1,PREFIX="train"):
      #return
      pass

    def get_categories(self):
        for path in os.listdir(self.PATH):
            if '.DS_Store' in path:
                pass
            else:
                self.list_categories.append(path)
        self.log("Found Categories ",self.list_categories,'\n')
        return self.list_categories

    def Process_Image(self):
        try:
            """
            Return Numpy array of image
            :return: X_Data, Y_Data
            """
            self.CATEGORIES = self.get_categories()
            for categories in self.CATEGORIES:                                                  # Iterate over categories

                train_folder_path = os.path.join(self.PATH, categories)                         # Folder Path
                class_index = self.CATEGORIES.index(categories)                                 # this will get index for classification

                for img in os.listdir(train_folder_path):                                       # This will iterate in the Folder
                    new_path = os.path.join(train_folder_path, img)                             # image Path

                    try:        # if any image is corrupted
                        image_data_temp = cv2.imread(new_path,cv2.IMREAD_GRAYSCALE)                 # Read Image as numbers
                        image_temp_resize = cv2.resize(image_data_temp,(self.IMAGE_SIZE,self.IMAGE_SIZE))
                        self.image_data.append([image_temp_resize,class_index])
                    except:
                        pass

            data = np.asanyarray(self.image_data)

            # Iterate over the Data
            for x in data:
                self.x_data.append(x[0])        # Get the X_Data
                self.y_data.append(x[1])        # get the label

            X_Data = np.asarray(self.x_data) / (255.0)      # Normalize Data
            Y_Data = np.asarray(self.y_data)

            # reshape x_Data

            X_Data = X_Data.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)

            return X_Data, Y_Data
        except:
            self.log("Failed to run Function Process Image ")

    def pickle_image(self):

        """
        :return: None Creates a Pickle Object of DataSet
        """
        # Call the Function and Get the Data
        X_Data,Y_Data = self.Process_Image()

        # Write the Entire Data into a Pickle File
        pickle_out = open('X_Data','wb')
        pickle.dump(X_Data, pickle_out)
        pickle_out.close()

        # Write the Y Label Data
        pickle_out = open('Y_Data', 'wb')
        pickle.dump(Y_Data, pickle_out)
        pickle_out.close()

        self.log("Pickled Image Successfully ")
        return X_Data,Y_Data

    def load_dataset(self):

        try:
            # Read the Data from Pickle Object
            X_Temp = open('X_Data','rb')
            X_Data = pickle.load(X_Temp)

            Y_Temp = open('Y_Data','rb')
            Y_Data = pickle.load(Y_Temp)

            self.log('Reading Dataset from PIckle Object')

            return X_Data,Y_Data

        except:
            self.log('Could not Found Pickle File ')
            self.log('Loading File and Dataset  ..........')

            X_Data,Y_Data = self.pickle_image()
            return X_Data,Y_Data

'''
#intended future usage
if __name__ == "__main__":
    
    path = 'path_to_your_training_dataset_folder' 
    
    im_ldr = ImageLoader(PATH=path,DIRECTORY_STRUCTURE=1,MAX_IMAGES_PER_CAT=100,IMAGE_SIZE=80)
    
    im_ldr.config_image(IMAGE_SIZE=80,FLATTERN=True,CUSTOM_PREPROCESS=function123)				
      
    im_ldr.save(SAVE_PATH='current_folder',PREFIX="train",SAVE_BATCH_SIZE=False) 
      
    X_Data,Y_Data = im_ldr.load(BATCH_NO=1,PREFIX="train") 

    print(X_Data.shape)

'''
