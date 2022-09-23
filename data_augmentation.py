import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
   

class DataAugmentation:

    def __init__(self, image_path, aug_path):

        self.image_path = image_path
        self.aug_path = aug_path
        self.datagen = ImageDataGenerator( 
                rotation_range = 40, 
                shear_range = 0.2, 
                zoom_range = 0.2, 
                horizontal_flip = True, 
                brightness_range = (0.5, 1.5)) 

    def aug_image(self): 

        path = self.image_path 
        path2 = self.aug_path  
        
        for i in os.listdir(path):
            try:
                img = load_img(path + i)  
                print(path + i)
                # Converting the input sample image to an array 
                x = img_to_array(img) 
                # Reshaping the input image 
                x = x.reshape((1, ) + x.shape)  
                
                # Generating and saving 5 augmented samples  
                # using the above defined parameters.  
                i = 0
                for batch in self.datagen.flow(x, batch_size = 1, 
                                            save_to_dir = path2,
                                            save_prefix ='image', save_format ='png'): 
                    i += 1
                    if i > 5: 
                        break
            except:
                continue