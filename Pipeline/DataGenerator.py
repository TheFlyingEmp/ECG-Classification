import keras
import numpy as np
CONFIDENCE_THRESHOLD: float = 0.5  

class IncepSE_ECG_Generator(keras.utils.Sequence):
    def __init__(self, dataframe, ecg_data, meta_data, binarizer, batch_size=32, 
                 shuffle=True, seed=42, confidence_threshold=0.5):
        self.dataframe = dataframe
        self.ecg_data = ecg_data
        self.meta_data = meta_data
        self.binarizer = binarizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.confidence_threshold = confidence_threshold
        self.indexes = np.arange(len(self.dataframe))
        self.epochs_completed = 0
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get ECG data
        batch_ecg = self.ecg_data[batch_indexes]
        
        # Get metadata
        batch_meta = self.meta_data[batch_indexes]
        
        # Get and binarize SCP codes with confidence filtering
        scp_codes = self.dataframe.iloc[batch_indexes]['scp_codes']
        scp_lists = []
        for code_dict in scp_codes:
            # Apply confidence threshold
            codes = [code for code, conf in code_dict.items() if conf >= self.confidence_threshold]
            scp_lists.append(codes)
        
        batch_labels = self.binarizer.transform(scp_lists)
        
        # Return both ECG data and metadata as inputs
        return (batch_ecg, batch_meta), batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epochs_completed)
            rng.shuffle(self.indexes)
        self.epochs_completed += 1

    def get_epoch_info(self):
        """Return information about completed epochs"""
        return {"epochs_completed": self.epochs_completed, "shuffle_seed": self.seed}