from keras import Model, layers

def IncepSE_Model(signal_input_shape, num_classes: int, num_meta_features: int):
    # Input for ECG signal: (time=1000, channels=12)
    sig_inp = layers.Input(shape=signal_input_shape, name="ecg_input")
    
    # Input for metadata
    meta_inp = layers.Input(shape=(num_meta_features,), name="metadata_input")
    
    # Process metadata with fully connected layers
    meta = layers.Dense(64, activation='relu', name="meta_fc1")(meta_inp)
    meta = layers.Dropout(0.2, name="meta_dropout1")(meta)
    meta = layers.Dense(32, activation='relu', name="meta_fc2")(meta)
    meta = layers.Dropout(0.2, name="meta_dropout2")(meta)
    
    # Project raw 12 -> 128 (do this once, outside the IncepSE blocks)
    proj_layer = layers.Conv1D(128, 1, padding='same', name="input_projector")(sig_inp)
    proj_layer = layers.BatchNormalization()(proj_layer)
    proj_layer = layers.ReLU()(proj_layer)

    # Stack six IncepSE blocks (internal filters=128)
    x = proj_layer
    for i in range(6):
        x = IncepSE_block(x, filters=128, bottleneck=32, name=f"incepse_{i}")

    # Final additional IncepSE: double channels to 256, downsample=True, dropout
    x = IncepSE_block(x, filters=256, bottleneck=64, downsample=True, dropout_rate=0.3, name="final_incepse")

    # Global average pool over time
    x = layers.GlobalAveragePooling1D()(x)  # (batch, 256)
    
    # Combine ECG features with metadata
    combined = layers.concatenate([x, meta], name="feature_fusion")
    
    # Additional fully connected layer after fusion
    combined = layers.Dense(128, activation='relu', name="fusion_fc1")(combined)
    combined = layers.Dropout(0.3, name="fusion_dropout")(combined)
    
    # Output layer
    out = layers.Dense(num_classes, activation='sigmoid', name="output")(combined)

    classifier = Model(inputs=[sig_inp, meta_inp], outputs=out, name="PLoS_Classifier")
    return classifier

def IncepSE_block(x,
                  filters=128,         # number of output channels of the block
                  bottleneck=32,       # channels inside bottleneck / each branch
                  kernel_sizes=(10, 20, 40),
                  pool_kernel=3,
                  downsample=False,
                  dropout_rate=0.0,
                  name=None):
    """
    x: input tensor (batch, time, channels)
    filters: number of output channels after concat/residual
    bottleneck: number of filters in the 1x1 bottleneck and each branch conv
    downsample: whether to downsample time length by factor 2
    returns: output tensor
    """
    stride = 2 if downsample else 1
    input_channels = x.shape[-1]
    
    # Bottleneck 1x1 -> then SE
    b0 = layers.Conv1D(bottleneck, 1, padding='same', 
                      name=f"{name}_b0_conv" if name else None)(x)
    b0 = SE_Block(b0, name=f"{name}_se" if name else None)

    # Branch: pooled -> conv
    b1 = layers.MaxPool1D(pool_size=pool_kernel, padding='same', strides=1, 
                         name=f"{name}_b1_pool" if name else None)(x)
    b1 = layers.Conv1D(bottleneck, 3, padding='same', strides=stride, 
                      name=f"{name}_b1_conv" if name else None)(b1)

    # Other branches from bottleneck
    b_list = [b1]
    for i, k in enumerate(kernel_sizes, start=2):
        br = layers.Conv1D(bottleneck, k, padding='same', strides=stride, 
                          name=f"{name}_b{i}_conv_k{k}" if name else None)(b0)
        b_list.append(br)

    # Concat
    concat = layers.concatenate(b_list, axis=-1, 
                               name=f"{name}_concat" if name else None)
    concat = layers.BatchNormalization(name=f"{name}_bn" if name else None)(concat)
    concat = layers.ReLU(name=f"{name}_relu" if name else None)(concat)

    if dropout_rate and dropout_rate > 0.0:
        concat = layers.Dropout(dropout_rate, name=f"{name}_drop" if name else None)(concat)

    # Residual connection if dimensions match, otherwise use projection
    if input_channels == filters and stride == 1:
        residual = x
    else:
        # Projection to match dimensions
        residual = layers.Conv1D(filters, 1, padding='same', strides=stride, 
                                name=f"{name}_res_conv" if name else None)(x)
        residual = layers.BatchNormalization(name=f"{name}_res_bn" if name else None)(residual)

    out = layers.add([concat, residual], name=f"{name}_add" if name else None)
    out = layers.ReLU(name=f"{name}_out_relu" if name else None)(out)
    return out

def SE_Block(input_tensor, reduction_ratio: int = 16, name: str = None):
    """Squeeze and Excitation block"""
    channels = input_tensor.shape[-1]
    
    # Squeeze (global average pooling)
    se = layers.GlobalAveragePooling1D(name=f"{name}_gap" if name else None)(input_tensor)
    se = layers.Reshape((1, channels), name=f"{name}_reshape" if name else None)(se)
    
    # Excitation
    se = layers.Dense(channels // reduction_ratio, activation='relu', 
                     name=f"{name}_se_fc1" if name else None)(se)
    se = layers.Dense(channels, activation='sigmoid', 
                     name=f"{name}_se_fc2" if name else None)(se)
    
    # Scale the input
    scaled = layers.multiply([input_tensor, se], 
                           name=f"{name}_se_scale" if name else None)
    return scaled