from __future__ import print_function, division
#import keras
from tensorflow import keras
from datetime import datetime


def train(args, preprocessor):
    preprocessor.set_training_set()

    features_data = preprocessor.data_structure['data']['train']['features_data']
    labels = preprocessor.data_structure['data']['train']['labels']
    max_length_process_instance = preprocessor.data_structure['meta']['max_length_process_instance']
    num_features = preprocessor.data_structure['meta']['num_features']
    num_event_ids = preprocessor.data_structure['meta']['num_event_ids']

    print('Create machine learning model ... \n')

    # Long short-term memory neural network
    if args.dnn_architecture == 0:
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')
        l1 = keras.layers.LSTM(100, implementation=2, activation="tanh", kernel_initializer='glorot_uniform',
                                         return_sequences=False, dropout=0.3)(main_input)
        b1 = keras.layers.BatchNormalization()(l1)

    #MLP
    elif  args.dnn_architecture ==1: # layer 1
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')
        input_layer_flattened = keras.layers.Flatten()(main_input)

        # layer 2
        hidden_layer_1 = keras.layers.Dense(300, activation='relu')(input_layer_flattened)
        hidden_layer_1 = keras.layers.BatchNormalization()(hidden_layer_1)
        hidden_layer_1 = keras.layers.Dropout(0.5)(hidden_layer_1)

        # layer 3
        hidden_layer_2 = keras.layers.Dense(200, activation='relu')(hidden_layer_1)
        hidden_layer_2 = keras.layers.BatchNormalization()(hidden_layer_2)
        hidden_layer_2 = keras.layers.Dropout(0.5)(hidden_layer_2)

        # layer 4
        hidden_layer_3 = keras.layers.Dense(100, activation='relu')(hidden_layer_2)
        hidden_layer_3 = keras.layers.BatchNormalization()(hidden_layer_3)
        hidden_layer_3 = keras.layers.Dropout(0.5)(hidden_layer_3)

        # layer 5
        hidden_layer_4 = keras.layers.Dense(50, activation='relu')(hidden_layer_3)
        hidden_layer_4 = keras.layers.BatchNormalization()(hidden_layer_4)
        hidden_layer_output = keras.layers.Dropout(0.5)(hidden_layer_4)

    elif args.dnn_architecture == 2:
        # layer 1
        main_input = keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')

        # layer 2
        hidden_layer_1 = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', strides=1, activation='relu')(
            main_input)
        hidden_layer_1 = keras.layers.MaxPool1D(data_format='channels_first')(hidden_layer_1)

        # layer 3
        hidden_layer_2 = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', strides=1, activation='relu')(
            hidden_layer_1)
        hidden_layer_2 = keras.layers.MaxPool1D(data_format='channels_first')(hidden_layer_2)

        # layer 4
        hidden_layer_3 = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', strides=1, activation='relu')(
            hidden_layer_2)
        hidden_layer_3 = keras.layers.MaxPool1D(data_format='channels_first')(hidden_layer_3)

        # layer 5
        hidden_layer_4 = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', strides=1, activation='relu')(
            hidden_layer_3)
        hidden_layer_4 = keras.layers.MaxPool1D(data_format='channels_first')(hidden_layer_4)

        # layer 6
        hidden_layer_5 = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', strides=1, activation='relu')(
            hidden_layer_4)
        hidden_layer_5 = keras.layers.MaxPool1D(data_format='channels_first')(hidden_layer_5)

        # layer 7
        hidden_layer_6 = keras.layers.Flatten()(hidden_layer_5)

        # layer 8
        hidden_layer_output = keras.layers.Dense(100, activation='relu')(hidden_layer_6)
        #hidden_layer_output = keras.layers.Dropout(0.5)(hidden_layer_4)


    if args.dnn_architecture == 0:
        act_output = keras.layers.Dense(num_event_ids, activation='softmax', name='act_output',
                                        kernel_initializer='glorot_uniform')(b1)
        model = keras.models.Model(inputs=[main_input], outputs=[act_output])
        optimizer = keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                       schedule_decay=0.004, clipvalue=3)

    # Note: this architectures does not use the nadam optimizer but the adam optimizer
    elif args.dnn_architecture == 1:
        act_output = keras.layers.Dense(num_event_ids, activation='softmax', name='act_output',
                                        kernel_initializer='glorot_uniform')(hidden_layer_output)
        model = keras.models.Model(inputs=[main_input], outputs=[act_output])
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    # Note: this architectures does not use the nadam optimizer but the adam optimizer
    elif args.dnn_architecture == 2:
        act_output = keras.layers.Dense(num_event_ids, activation='softmax', name='act_output',
                                        kernel_initializer='glorot_uniform')(hidden_layer_output)
        model = keras.models.Model(inputs=[main_input], outputs=[act_output])
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9,
                                          beta_2=0.999)

    model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = keras.callbacks.ModelCheckpoint('%smodel_%s.h5' % (args.checkpoint_dir,
                                                                          preprocessor.data_structure['support'][
                                                                              'iteration_cross_validation']),
                                                       monitor='val_loss',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto')
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                                   min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()

    start_training_time = datetime.now()

    model.fit(features_data, {'act_output': labels}, validation_split=1 / args.num_folds, verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)

    training_time = datetime.now() - start_training_time

    return training_time.total_seconds()
