from cdisc_kutils import *
from cdisc_models import *
import glob
import functools
from keras import layers
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import plot_model
from keras.optimizers import Adam

#plot_model(model, to_file='model.png')
pd.options.display.max_rows = 999
seed = 2017
np.set_printoptions(threshold=np.inf, precision=4)
np.random.seed(seed)



if __name__ == '__main__':
    # data loadng    
    train_offsets_df = pd.read_pickle(os.path.join(data_dir, "whole_offset.pkl"))
    test_offsets_df = pd.read_pickle(os.path.join(data_dir, "test_offsets.pkl"))
    train_images_df = pd.read_pickle(os.path.join(data_dir, "train_offsets_0.9.pkl"))
    valid_images_df = pd.read_pickle(os.path.join(data_dir, "valid_offsets_0.1.pkl"))
    test_images_df = pd.read_pickle(os.path.join(data_dir, "test_imgs_df.pkl"))
    cat2idx = pickle.load(open(os.path.join(data_dir, "cat2idx.pkl"), "rb")) 
    idx2cat = pickle.load(open(os.path.join(data_dir, "idx2cat.pkl"), "rb"))
    train_bson_file = open(os.path.join(data_dir, "train.bson"), "rb")
    test_bson_file = open(os.path.join(data_dir, "test.bson"), "rb")
    
    #parameters
    num_classes = len(cat2idx)
    img_size = 180
    batch_size = 256
    
    
    
    
    # model topology
    model_base = Models(img_size, num_classes)
    resnet_model = model_base.resnet50()
    finetune_from(resnet_model, finetune_layer='res5b_branch2a')
    resnet_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        
    lock = threading.Lock()
    # generators
    #train_datagen = ImageDataGenerator(horizontal_flip=True, height_shift_range=0.3, width_shift_range=0.3)
    train_datagen = ImageDataGenerator()
    train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df, 
                         num_classes, train_datagen, lock, target_size=(img_size, img_size),
                         batch_size=batch_size, shuffle=True)
    
    valid_datagen = ImageDataGenerator()
    valid_gen = BSONIterator(train_bson_file, valid_images_df, train_offsets_df, 
                         num_classes, valid_datagen, lock, target_size=(img_size, img_size),
                         batch_size=batch_size, shuffle=False)
    
    
    # epoch 1
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    
    resnet_model.save("../data/cdisc/models/ResNet50/{}_val_acc_{:.4f}_epoch1.h5".format('res5b_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 2
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    
    resnet_model.save("../data/cdisc/models/ResNet50/{}_val_acc_{:.4f}_epoch2.h5".format('res5b_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 3
    finetune_from(resnet_model, finetune_layer='res5a_branch2a')
    resnet_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    
    resnet_model.save("../data/cdisc/models/ResNet50/{}_val_acc_{:.4f}_epoch3.h5".format('res5a_branch2a', history.history['val_acc'][0]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # epoch 1
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    
    resnet_model.save("../data/cdisc/models/ResNet50/{}_val_acc_{:.4f}_epoch1.h5".format('res5c_branch2a', history.history['val_acc'][0]))
    
    # epoch 2
    finetune_from(resnet_model, finetune_layer='res5b_branch2a')
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    
    resnet_model.save("../data/cdisc/models/ResNet50/{}_val_acc_{:.4f}_epoch2.h5".format('res5b_branch2a', history.history['val_acc'][0]))
    
    # epoch 3
    finetune_from(resnet_model, finetune_layer='res5a_branch2a')
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    
    resnet_model.save("../data/cdisc/models/ResNet50/{}_val_acc_{:.4f}_epoch3.h5".format('res5a_branch2a', history.history['val_acc'][0]))
    
    