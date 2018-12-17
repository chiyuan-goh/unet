from model import *
from data import *
from time import time
import sys
from keras.callbacks import TensorBoard
from utils import MaskImageTensorBoard

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
	model_name = 'unet_apollo.hdf5'

	if sys.argv[1] ==  'train':
		data_gen_args = dict(rotation_range=0.2,
		                    width_shift_range=0.05,
		                    height_shift_range=0.05,
		                    shear_range=0.05,
		                    zoom_range=0.05,
		                    horizontal_flip=True,
		                    fill_mode='nearest')
		
		myGene = trainGenerator(4,'data/apollo/train','image','label',data_gen_args,save_to_dir = 'data/apollo/tmp', image_color_mode='rgb')
		myVal = trainGenerator(4,'data/apollo/train','image','label',data_gen_args,save_to_dir = 'data/apollo/tmp', image_color_mode='rgb')
		
		model = unet(input_size = (256, 256, 3))

		model_checkpoint = ModelCheckpoint(model_name, monitor='loss',verbose=1, save_best_only=True)
		tboard = MaskImageTensorBoard(image_every_x_epochs=10, output_name='conv2d_24', log_dir="logs/{}".format(time()), histogram_freq=1)
		#tboard = TensorBoard(log_dir="logs/{}".format(time()), update_freq = 'epoch')

		#model.fit_generator(myGene,steps_per_epoch=150,epochs=10,callbacks=[model_checkpoint, tboard], validation_data=myVal, validation_steps=2)
		model.fit_generator(myGene,steps_per_epoch=10,epochs=10,callbacks=[model_checkpoint, tboard], validation_data=myVal, validation_steps=2)
	else:
		model = unet(pretrained_weights=model_name, input_size = (256, 256, 3))


	testGene = testGenerator("data/apollo/test", num_image=19, as_gray=False)
	results = model.predict_generator(testGene,19,verbose=1)
	saveResult("data/apollo/test",results)