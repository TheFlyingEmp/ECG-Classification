import os
import pandas as pd
import numpy as np
import wfdb
from scipy import signal
from DataGenerator import IncepSE_ECG_Generator
from PreprocessDataset import PreprocessData
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from Model import IncepSE_Model
from keras import optimizers, losses
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import SpecificityAtSensitivity, AUC, BinaryAccuracy, Precision, Recall
from keras.saving import save_model
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Hyperparameters
EPOCHS: int = 30
ONE_CYCLE_EPOCHS: int = 13
LR: float = 1e-2
BATCH_SIZE: int = 64
SAMPLING_RATE: int = 100
GLOBAL_SEED: int = 42
CONFIDENCE_THRESHOLD: float = 0.5

# Butterworth bandpass filter implementation
def _butterworth_bandpass_filter(data, lowcut=1.0, highcut=45.0, fs=100, order=4):
    """
    Apply a Butterworth bandpass filter to ECG data
    
    Parameters:
    data: numpy array of shape (n_samples, n_timesteps, n_channels)
    lowcut: Low cutoff frequency (Hz)
    highcut: High cutoff frequency (Hz)
    fs: Sampling frequency (Hz)
    order: Filter order
    
    Returns:
    Filtered ECG data
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter to each channel of each sample
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):  # For each sample
        for j in range(data.shape[2]):  # For each channel
            filtered_data[i, :, j] = signal.filtfilt(b, a, data[i, :, j])
    
    return filtered_data

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    
    # Normalize each channel to zero mean and unit variance
    for i in range(data.shape[0]):  # For each sample
        for j in range(data.shape[2]):  # For each channel
            channel_data = data[i, :, j]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:  # Avoid division by zero
                data[i, :, j] = (channel_data - mean) / std
            else:
                data[i, :, j] = 0  # Set to zero if no variance
    
    # Apply Butterworth bandpass filter (1-45 Hz) for 100Hz data
    if sampling_rate == 100:
        data = _butterworth_bandpass_filter(data, lowcut=1.0, highcut=45.0, fs=100, order=4)
    
    return data

def filter_examples_with_confident_labels(df, ecg_data, meta_data, confidence_threshold):
    """
    Filter examples to only include those with at least one confident SCP code
    Returns filtered DataFrame, ECG data, and metadata
    """
    indices_to_keep = []
    for idx, code_dict in enumerate(df['scp_codes']):
        codes = [code for code, conf in code_dict.items() if conf >= confidence_threshold]
        if len(codes) > 0:
            indices_to_keep.append(idx)
    
    filtered_df = df.iloc[indices_to_keep].reset_index(drop=True)
    filtered_ecg = ecg_data[indices_to_keep]
    filtered_meta = meta_data[indices_to_keep] if meta_data is not None else None
    
    print(f"Filtered out {len(df) - len(filtered_df)} examples with no confident labels")
    return filtered_df, filtered_ecg, filtered_meta

def get_full_predictions(model, generator):
    """
    Collect full y_true and y_pred from a generator.
    Assumes generator yields ((ecg, meta), y)
    """
    y_true_list = []
    y_pred_list = []
    generator.shuffle = False  # Ensure deterministic
    for batch in generator:
        if batch is None:
            break
        (ecg, meta), y = batch
        pred = model.predict((ecg, meta), verbose=0)
        y_true_list.append(y)
        y_pred_list.append(pred)
    return np.vstack(y_true_list), np.vstack(y_pred_list)

def plot_history(history):
    """Plot training history and save as PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # AUC
    axes[0, 1].plot(history.history['auc'], label='Train AUC')
    axes[0, 1].plot(history.history['val_auc'], label='Val AUC')
    axes[0, 1].set_title('Model AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('classifier_history_plot.png')
    plt.close()
    print("History plot saved as 'classifier_history_plot.png'")

def compute_and_plot_confusion_matrix(y_true, y_pred, dataset_name, threshold=0.5):
    """Compute and plot overall confusion matrix (flattened multi-label)."""
    y_true_flat = y_true.ravel()
    y_pred_bin = (y_pred > threshold).ravel().astype(int)
    
    cm = confusion_matrix(y_true_flat, y_pred_bin)
    print(f"\n{dataset_name} Confusion Matrix (Overall Flattened) at threshold {threshold}:\n", cm)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{dataset_name} Confusion Matrix (Threshold: {threshold})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{dataset_name.lower().replace(" ", "_")}_confusion_matrix_thresh_{threshold}.png')
    plt.close()
    print(f"{dataset_name} confusion matrix plot saved as '{dataset_name.lower().replace(' ', '_')}_confusion_matrix_thresh_{threshold}.png'")

def find_diagnostic_threshold(y_true, y_pred, method='youden'):
    """
    Compute optimal threshold for diagnostic tool from ROC curve.
    For diagnostic: Maximize Youden's J (sensitivity + specificity - 1) for balanced high specificity/precision.
    Flattens multi-label for global threshold.
    """
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    
    fpr, tpr, thresholds = roc_curve(y_true_flat, y_pred_flat)
    roc_auc = auc(fpr, tpr)
    
    if method == 'youden':
        # Youden's J for diagnostic balance (high spec + sens)
        youden = tpr - fpr
        optimal_idx = np.argmax(youden)
    elif method == 'closest_topleft':
        # Closest to (0,1) - favors high specificity
        distances = np.sqrt(fpr**2 + (1 - tpr)**2)
        optimal_idx = np.argmin(distances)
    else:
        raise ValueError("Method must be 'youden' or 'closest_topleft'")
    
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_specificity = 1 - optimal_fpr
    
    # Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(optimal_fpr, optimal_tpr, color='red', s=100, 
                label=f'Optimal (Spec={optimal_specificity:.3f}, Sens={optimal_tpr:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Diagnostic Threshold Selection ({method})')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_diagnostic.png')
    plt.close()
    print(f"ROC plot saved as 'roc_curve_diagnostic.png'")
    
    print(f"Optimal diagnostic threshold ({method}): {optimal_threshold:.3f}")
    print(f"Associated metrics: Sensitivity={optimal_tpr:.3f}, Specificity={optimal_specificity:.3f}")
    
    return optimal_threshold

def Train(Root_Dir: str, DF_Path=None):
    DATA_DIR = os.path.join(Root_Dir, "Dataset", "PTB-XL")

    FinalDF = None
    
    # ------------- Load & prepare dataframe -------------
    try:
        FinalDF = pd.read_csv(DF_Path, index_col='ecg_id')
        FinalDF["scp_codes"] = FinalDF["scp_codes"].apply(lambda x: ast.literal_eval(x))
    except:
        FinalDF = PreprocessData(Root_Dir)

    # Get meta features (excluding non-meta columns)
    non_meta_cols = ["patient_id", "scp_codes", "strat_fold", "filename_lr", "filename_hr"]
    meta_cols = [col for col in FinalDF.columns if col not in non_meta_cols]
    NUM_META_FEATURES = len(meta_cols)
    
    # Split data - note the parentheses around conditions
    Train_DF = FinalDF[(FinalDF["strat_fold"] != 10) & (FinalDF["strat_fold"] != 9)]
    Val_DF = FinalDF[FinalDF["strat_fold"] == 9]
    Test_DF = FinalDF[FinalDF["strat_fold"] == 10]

    # Load ECG data for each split separately
    Train_ECG = load_raw_data(df=Train_DF, sampling_rate=SAMPLING_RATE, path=DATA_DIR)
    Val_ECG = load_raw_data(df=Val_DF, sampling_rate=SAMPLING_RATE, path=DATA_DIR)
    Test_ECG = load_raw_data(df=Test_DF, sampling_rate=SAMPLING_RATE, path=DATA_DIR)

    # Extract metadata for each split
    Train_Meta = Train_DF[meta_cols].values
    Val_Meta = Val_DF[meta_cols].values
    Test_Meta = Test_DF[meta_cols].values

    # Fit binarizer with confidence filtering on training data only
    all_train_scp_codes = []
    for code_dict in Train_DF['scp_codes']:
        codes = [code for code, conf in code_dict.items() if conf >= CONFIDENCE_THRESHOLD]
        all_train_scp_codes.extend(codes)

    unique_codes = sorted(set(all_train_scp_codes))
    BinarizerInst = MultiLabelBinarizer(classes=unique_codes).fit(y= unique_codes)

    joblib.dump(BinarizerInst, "Multi_Label_Binarizer_Inst.joblib")
    
    # Filter training examples to only include those with confident labels
    Train_DF, Train_ECG, Train_Meta = filter_examples_with_confident_labels(
        Train_DF, Train_ECG, Train_Meta, CONFIDENCE_THRESHOLD
    )
    
    # For validation and test sets, we keep all examples but will handle no-label cases in the generator
    # This ensures we evaluate on the full dataset while training only on confident examples

    # Create generators with both ECG and metadata
    TrainGen = IncepSE_ECG_Generator(
        dataframe=Train_DF, 
        ecg_data=Train_ECG,
        meta_data=Train_Meta,
        binarizer=BinarizerInst, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        seed=GLOBAL_SEED,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    ValGen = IncepSE_ECG_Generator(
        dataframe=Val_DF, 
        ecg_data=Val_ECG,
        meta_data=Val_Meta,
        binarizer=BinarizerInst, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        seed=GLOBAL_SEED,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    TestGen = IncepSE_ECG_Generator(
        dataframe=Test_DF, 
        ecg_data=Test_ECG,
        meta_data=Test_Meta,
        binarizer=BinarizerInst, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        seed=GLOBAL_SEED,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    print(">>> Starting Classifier Training")
    print(f"Training on {len(Train_DF)} examples with confident labels")
    print(f"Validating on {len(Val_DF)} examples (some may have no confident labels)")
    print(f"Testing on {len(Test_DF)} examples (some may have no confident labels)")

    # Define signal input shape based on your ECG data
    # Assuming shape: (time_steps, channels) = (1000, 12)
    signal_input_shape = (1000, 12)
    
    ClassifierInst = IncepSE_Model(
        signal_input_shape=signal_input_shape, 
        num_classes=len(unique_codes),  # Use number of classes from training data
        num_meta_features=NUM_META_FEATURES
    )

    ClassifierInst.compile(
        optimizer=optimizers.AdamW(learning_rate= LR, weight_decay=1e-4, clipvalue=0.1),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            BinaryAccuracy(),
            AUC(name="auc", multi_label=True, num_thresholds=200),
            Precision(name="precision"),
            Recall(name="recall"),
            SpecificityAtSensitivity(0.5, name="specificity_at_sensitivity")
        ]
    )
    
    # Calculate steps for OneCycle
    steps_per_epoch = int(np.ceil(len(Train_DF) / BATCH_SIZE))
    total_steps = ONE_CYCLE_EPOCHS * steps_per_epoch
    warmup_steps = int(total_steps * 0.1)  # 10% warmup

    val_steps_cls = int(np.ceil(len(Val_DF) / BATCH_SIZE))

    # Create OneCycle scheduler (without auto-stop)
    onecycle_cb = OneCycleScheduler(
        max_lr=LR,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        div_factor=25.,
        final_div_factor=1e4,
        stop_after_cycles=False  # Don't stop automatically
    )

    # Create ReduceLROnPlateau for fine-tuning
    reduce_lr = ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6,
        mode='max'
    )

    # Create hybrid scheduler
    hybrid_scheduler = HybridSchedulerCallback(
        onecycle_cb=onecycle_cb,
        reduce_lr_cb=reduce_lr,
        switch_epoch=ONE_CYCLE_EPOCHS  # Switch after OneCycle completes
    )
    
    # Additional callbacks
    checkpoint_cb = ModelCheckpoint(
        'classifier_best_cpkt.keras', 
        save_best_only=True, 
        monitor='val_auc',  # Monitor AUC as in the paper
        mode='max'
    )
    
    early_stop_cb = EarlyStopping(
        patience=12,
        monitor='val_auc',
        mode='max',
        restore_best_weights=True
    )

    # Update your callbacks list:
    callbacks=[
        hybrid_scheduler,
        checkpoint_cb, 
        early_stop_cb
    ]
    
    # Train the model with OneCycle
    history = ClassifierInst.fit(
        TrainGen,
        validation_data=ValGen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps_cls,
        verbose=1,
        callbacks= callbacks
    )
    
    save_model(model= ClassifierInst, filepath= "Classifier_Last_Epoch.keras")
    pd.DataFrame(history.history).to_csv(f"ClassifierHist.csv", index=False)
    
    # Plot training history
    plot_history(history)
    
    # Metric names for evaluation
    metric_names = ['loss', 'binary_accuracy', 'auc', 'precision', 'recall', 'specificity_at_sensitivity']
    
    # Evaluate on train, val, test
    train_results = ClassifierInst.evaluate(TrainGen, verbose=1)
    val_results = ClassifierInst.evaluate(ValGen, verbose=1)
    test_results = ClassifierInst.evaluate(TestGen, verbose=1)
    
    # Print metrics in a table
    results_df = pd.DataFrame({
        'Metric': metric_names,
        'Train': [f"{val:.4f}" for val in train_results],
        'Validation': [f"{val:.4f}" for val in val_results],
        'Test': [f"{val:.4f}" for val in test_results]
    })
    print("\nFinal Evaluation Metrics:")
    print(results_df.to_string(index=False))
    
    # Get predictions for confusion matrices (on test only for brevity; extend if needed)
    print("\nComputing predictions for confusion matrix...")
    test_y_true, test_y_pred = get_full_predictions(ClassifierInst, TestGen)
    
    # Default confusion matrix at 0.5
    compute_and_plot_confusion_matrix(test_y_true, test_y_pred, "Test Set", threshold=0.5)
    
    # Find diagnostic threshold from ROC
    optimal_thresh = find_diagnostic_threshold(test_y_true, test_y_pred, method='youden')
    
    # Recompute confusion matrix and metrics at optimal threshold
    compute_and_plot_confusion_matrix(test_y_true, test_y_pred, "Test Set Optimal", threshold=optimal_thresh)
    
    # Compute macro-averaged metrics at optimal threshold
    y_pred_bin_opt = (test_y_pred > optimal_thresh).astype(int)
    precision_opt = precision_score(test_y_true, y_pred_bin_opt, average='macro', zero_division=0)
    recall_opt = recall_score(test_y_true, y_pred_bin_opt, average='macro', zero_division=0)
    f1_opt = f1_score(test_y_true, y_pred_bin_opt, average='macro', zero_division=0)
    
    print(f"\nMetrics at optimal diagnostic threshold {optimal_thresh:.3f} (macro-averaged):")
    print(f"Precision: {precision_opt:.4f}, Recall: {recall_opt:.4f}, F1: {f1_opt:.4f}")

class OneCycleScheduler(Callback):
    def __init__(self, max_lr, total_steps, warmup_steps, div_factor=25., final_div_factor=1e4, stop_after_cycles=True):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.anneal_steps = total_steps - warmup_steps
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.stop_after_cycles = stop_after_cycles
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.step_count = 0
        self.lr_history = []
        
    def on_train_begin(self, logs=None):
        self.step_count = 0
        self.lr_history = []
        if hasattr(self, 'model') and self.model is not None:
            self.model.optimizer.learning_rate.assign(self.initial_lr)
        
    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self, 'model') or self.model is None:
            return
            
        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (
                self.step_count / self.warmup_steps)
        elif self.step_count < self.total_steps:
            # Cosine annealing
            p = (self.step_count - self.warmup_steps) / self.anneal_steps
            lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (
                1 + np.cos(np.pi * p))
        else:
            # After OneCycle completes, maintain final LR
            lr = self.final_lr
            
        lr = max(lr, self.final_lr)
        self.model.optimizer.learning_rate.assign(float(lr))
        self.lr_history.append(float(lr))
        self.step_count += 1
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if hasattr(self, 'model') and self.model is not None:
            logs['learning_rate'] = self.model.optimizer.learning_rate.numpy()
        
        # Stop training after OneCycle if requested
        if self.stop_after_cycles and epoch + 1 == ONE_CYCLE_EPOCHS:
            if hasattr(self, 'model') and self.model is not None:
                self.model.stop_training = True
            print(f"OneCycle completed at epoch {epoch+1}, ReduceLROnPlateau will take over")

class HybridSchedulerCallback(Callback):
    def __init__(self, onecycle_cb, reduce_lr_cb, switch_epoch):
        super().__init__()
        self.onecycle_cb = onecycle_cb
        self.reduce_lr_cb = reduce_lr_cb
        self.switch_epoch = switch_epoch
        self.current_scheduler = onecycle_cb
        
    def set_model(self, model):
        self.onecycle_cb.set_model(model)
        self.reduce_lr_cb.set_model(model)
        super().set_model(model)
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.switch_epoch:
            print(f"Switching from OneCycle to ReduceLROnPlateau at epoch {epoch+1}")
            self.current_scheduler = self.reduce_lr_cb
            
    def on_batch_begin(self, batch, logs=None):
        self.current_scheduler.on_batch_begin(batch, logs)
        
    def on_epoch_end(self, epoch, logs=None):
        self.current_scheduler.on_epoch_end(epoch, logs)