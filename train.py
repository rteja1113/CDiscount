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
    batch_size = 128
    
    
    
    
    # model topology
    model_base = Models(img_size, num_classes)
    resnet_model = model_base.resnet101()
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
    
    print(y)



    # warm-up
    #history = resnet_model.fit_generator(train_gen, steps_per_epoch=1000,
    #                                     validation_data=valid_gen,
    #                                     validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
    #                                     epochs=1, workers=8)
    
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch1.h5".format('res5c_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 2
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch2.h5".format('res5c_branch2a', history.history['val_acc'][0]))
    
    # epoch 3
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch3.h5".format('res5c_branch2a', history.history['val_acc'][0]))
    
    # epoch 4
    finetune_from(resnet_model, finetune_layer='res5a_branch2a')
    resnet_model.compile(Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch4.h5".format('res5c_branch2a', history.history['val_acc'][0]))
    
    # epoch 5
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch5.h5".format('res5c_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 6
    finetune_from(resnet_model, finetune_layer='res4b22_branch2a')
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch6.h5".format('res4b22_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 7
    finetune_from(resnet_model, finetune_layer='res4b21_branch2a')
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch7.h5".format('res4b21_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 8
    finetune_from(resnet_model, finetune_layer='res4b20_branch2a')
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch8.h5".format('res4b20_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 9
    finetune_from(resnet_model, finetune_layer='res4b18_branch2a')
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch9.h5".format('res4b18_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 10
    batch_size = 256
    finetune_from(resnet_model, finetune_layer='res4b17_branch2a')
    resnet_model.load_weights('../data/cdisc/models/resnet101/res4b18_branch2a_val_acc_0.6562_epoch9.h5')
    resnet_model.compile(Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch10.h5".format('res4b17_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 11
    batch_size = 256
    finetune_from(resnet_model, finetune_layer='res4b16_branch2a')
    resnet_model.compile(Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch11.h5".format('res4b16_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 12
    resnet_model.load_weights('../data/cdisc/models/resnet101/res4b16_branch2a_val_acc_0.6641_epoch11.h5')

    finetune_from(resnet_model, finetune_layer='res4b15_branch2a')
    resnet_model.compile(Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch12.h5".format('res4b15_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 13
    finetune_from(resnet_model, finetune_layer='res4b14_branch2a')
    resnet_model.compile(Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch13.h5".format('res4b14_branch2a', history.history['val_acc'][0]))
    
    
    # epoch 14
    finetune_from(resnet_model, finetune_layer='res4b12_branch2a')
    resnet_model.compile(Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    resnet_model.save("../data/cdisc/models/resnet101/{}_val_acc_{:.4f}_epoch14.h5".format('res4b12_branch2a', history.history['val_acc'][0]))
    
    
    
    
    
    
    
    
    
    ## finetune point
    finetune_from(resnet_model, finetune_layer='res5b_branch2a')
    history = resnet_model.fit_generator(train_gen, steps_per_epoch=math.ceil(train_images_df.shape[0]/batch_size),
                                         validation_data=valid_gen,
                                         validation_steps=math.ceil(valid_images_df.shape[0]/batch_size),
                                         epochs=1, workers=8)
    
    resnet_model.save('../data/cdisc/models/resnet101/val_acc_{}_epoch1'.format(round(history.history['val_acc'], 4)))

    













    
    