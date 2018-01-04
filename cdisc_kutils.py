import os, sys, math, io, warnings, gc
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson
import struct
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import *
import threading
import pickle
import re
import time
# backend
import tensorflow as tf

# keras libs
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.data_utils import OrderedEnqueuer, GeneratorEnqueuer, Sequence
from keras.utils.generic_utils import Progbar
from keras.applications.imagenet_utils import preprocess_input
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data_dir = "/media/cvpr/ssd/cdisc"
num_train_products = 7069896
num_test_products = 1768182




train_bson_path = os.path.join(data_dir, "train.bson")
test_bson_path = os.path.join(data_dir, "test.bson")

def make_category_file(category_names_file):
    categories_path = os.path.join(data_dir, category_names_file)
    categories_df = pd.read_csv(categories_path, index_col="category_id")
    
    # Maps the category_id to an integer index. This is what we'll use to
    # one-hot encode the labels.
    categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)
    categories_df.to_pickle(os.path.join(data_dir, "categories.pkl"))
    

def make_category_tables(categories_df, data_dir):
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
        
    pickle.dump(cat2idx, open(os.path.join(data_dir, "cat2idx.pkl"), "wb"))
    pickle.dump(idx2cat, open(os.path.join(data_dir, "idx2cat.pkl"), "wb"))
    



#cat2idx, idx2cat = make_category_tables()


def make_bson_file(bson_path, num_records, with_categories, filename):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    df.to_pickle(os.path.join(data_dir, "{}.pkl".format(filename)))
    


def make_val_set(df, train_filename, valid_filename, cat2idx=None, split_percentage=0.2, drop_percentage=0.0):
    # Find the product_ids for each category.
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])

    train_list = []
    val_list = []
    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]

            # Randomly remove products to make the dataset smaller.
            keep_size = int(len(product_ids) * (1. - drop_percentage))
            if keep_size < len(product_ids):
                product_ids = np.random.choice(product_ids, keep_size, replace=False)

            # Randomly choose the products that become part of the validation set.
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(product_ids, val_size, replace=False)
            else:
                val_ids = []

            # Create a new row for each image.
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, "num_imgs"]):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()
                
    columns = ["product_id", "category_idx", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)
    train_df.to_pickle(os.path.join(data_dir, "{}.pkl".format(train_filename)))
    if split_percentage > 0:
        val_df.to_pickle(os.path.join(data_dir, "{}.pkl".format(valid_filename)))



def make_test_set(df, test_filename):
    test_list = []
    with tqdm(total = len(df)) as pbar:
        for index, row in df.iterrows():
            r = [index]
            for img_idx in range(row["num_imgs"]):
                test_list.append(r + [img_idx])
            pbar.update()
    
    columns = ["product_id", "img_idx"]
    test_df = pd.DataFrame(test_list, columns=columns)
    test_df.to_pickle(os.path.join(data_dir, "{}.pkl".format(test_filename)))















class BSONIterator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, lock, target_size=(180, 180), 
                 with_labels=True, batch_size=32, shuffle=False, seed=None):

        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)

        print("Found %d images belonging to %d classes." % (self.samples, self.num_class))

        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        self.lock = lock

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_product_ids = np.zeros(len(index_array), dtype=np.int64)
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])

            # Grab the image from the product.
            item = bson.BSON.decode(item_data)
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]

            # Load the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)

            # Preprocess the image.
            x = img_to_array(img)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            
            #### IMPORTANT !!! BELOW ONLY FOR MIHA'S MODELS
            x = preprocess_input(x, mode='tf')
            
            
            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            batch_product_ids[i] = product_id
            if self.with_labels:
                batch_y[i, image_row["category_idx"]] = 1

        if self.with_labels:
            return batch_x, batch_y
        else:
            return batch_x, batch_product_ids

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)[0]
        return self._get_batches_of_transformed_samples(index_array)






def get_layer_idx_by_name(model, name):
    for i, layer in enumerate(model.layers):
        if layer.name == name:
            return i
        

def finetune_from(model, finetune_layer="res5b_branch2a"):
    finetune_layer_idx = get_layer_idx_by_name(model, finetune_layer)
    for i, layer in enumerate(model.layers):
        if i < finetune_layer_idx:
            layer.trainable = False
        else:
            layer.trainable = True
    return


def _make_predict_function(model):
    if not hasattr(model, 'predict_function'):
        model.predict_function = None
    if model.predict_function is None:
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs = model._feed_inputs + [K.learning_phase()]
        else:
            inputs = model._feed_inputs
            # Gets network outputs. Does not update weights.
            # Does update the network states.
        kwargs = getattr(model, '_function_kwargs', {})
        model.predict_function = K.function(inputs,
                                           model.outputs,
                                           updates=model.state_updates,
                                           name='predict_function',
                                           **kwargs)
        

def parse_row(a):
    a_str = np.array2string(a, max_line_width=np.inf).replace('[', '').replace(']', '')
    return re.sub(' +', ',', a_str)

def predict_generator(models, generator, steps,out_file,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      verbose=0):
    """Generates predictions for the input samples from a data generator.
    The generator should return the same kind of data as accepted by
    `predict_on_batch`.
    # Arguments
        generator: Generator yielding batches of input samples
                or an instance of Sequence (keras.utils.Sequence)
                object in order to avoid duplicate data
                when using multiprocessing.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_queue_size: Maximum size for the generator queue.
        workers: Maximum number of processes to spin up
            when using process based threading
        use_multiprocessing: If `True`, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        verbose: verbosity mode, 0 or 1.
    # Returns
        Numpy array(s) of predictions.
    # Raises
        ValueError: In case the generator yields
            data in an invalid format.
    """
    for model in models:_make_predict_function(model)
    
    steps_done = 0
    wait_time = 0.01
    score_dict = {}
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    enqueuer = None
    
    try:
        if is_sequence:
            enqueuer = OrderedEnqueuer(generator,
                                       use_multiprocessing=use_multiprocessing)
        else:
            enqueuer = GeneratorEnqueuer(generator,
                                         use_multiprocessing=use_multiprocessing,
                                         wait_time=wait_time)
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()

        if verbose == 1:
            progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = next(output_generator)
            if isinstance(generator_output, tuple):
                # Compatibility with the generators
                # used for training.
                if len(generator_output) == 2:
                    x, batch_product_ids = generator_output
                elif len(generator_output) == 3:
                    x, _, _ = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
            else:
                # Assumes a generator that only
                # yields inputs (not targets and sample weights).
                x = generator_output
            
            # batch prediction and appending to chunk    
            
            ### IIMPORTANT BELOW MODIFIED FOR MIHA MODELS
            for i, model in enumerate(models):
                mini_batch = x.copy()
                if i == 0:
                    mini_batch = ((mini_batch / 255.) - 0.5) * 2
                elif i == 1:
                    mini_batch = mini_batch / 255.
                else:
                    mini_batch = preprocess_input(mini_batch, mode='tf')
                
                outs = model.predict_on_batch(mini_batch)
                outs_flipped = model.predict_on_batch(mini_batch[:, :, ::-1, :])
                
                for i, p_id in enumerate(batch_product_ids):
                    if p_id not in score_dict:
                        score_dict[p_id] = []
                        score_dict[p_id].append(np.expand_dims(outs[i], axis=0))
                        score_dict[p_id].append(np.expand_dims(outs_flipped[i], axis=0))
                    else:
                        score_dict[p_id].append(np.expand_dims(outs[i], axis=0))
                        score_dict[p_id].append(np.expand_dims(outs_flipped[i], axis=0))
            
            # process preds once in a while
            if steps_done % 250 == 249:
                for key, value in score_dict.items():
                    if isinstance(value, list):
                        if len(value) == generator.offsets_df.loc[key, "num_imgs"]*len(models)*2:
                            score_dict[key] = np.concatenate(value, axis=0).mean(axis=0).argmax()
                
                
        
            
            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)
        
        # last chunk
        for key, value in score_dict.items():
            if isinstance(value, list):
                if len(value) == generator.offsets_df.loc[key, "num_imgs"]*len(models)*2:
                    score_dict[key] = np.concatenate(value, axis=0).mean(axis=0).argmax()    
                        

            
        
    finally:
        if enqueuer is not None:
            enqueuer.stop()
    
    return score_dict









