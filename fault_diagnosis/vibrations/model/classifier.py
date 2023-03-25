import tensorflow as tf


def add_block(model: tf.keras.Model, filters: int, kernel_size: int, max_pool: bool, batch_norm: bool) -> tf.keras.Model:
    if max_pool:
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same'))
        model.add(tf.keras.layers.MaxPool1D())
    else:
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=2, padding='same'))
    if batch_norm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    
    return model
    

def get_model(
        input_width: int, 
        out_classes: int, 
        conv_blocks: int, 
        init_filters: int, 
        kernel_size: int, 
        max_pool: bool, 
        batch_norm: bool, 
        dropout: bool,
) -> tf.keras.Model:
    model = tf.keras.Sequential([])
    
    # Model input
    model.add(tf.keras.layers.Input((input_width, 3)))
    
    # Conv blocks
    for i in range(conv_blocks):
        model = add_block(model, init_filters*(i+1), kernel_size, max_pool, batch_norm)
    
    # Model output
    model.add(tf.keras.layers.Conv1D(filters=init_filters*(conv_blocks+1), kernel_size=kernel_size, padding='same'))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    if dropout:
        model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(out_classes, activation='softmax'))

    return model
