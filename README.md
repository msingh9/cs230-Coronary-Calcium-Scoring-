This code is produced during a CS230 class project titled as "Prediction of Coronary Artery Disease Via Calcium Scoring
of Chest CTs". Code based can be used to train and evaluate models on "Stanford AIMI COCA dataset 15, which was collected by Stanford Hospital and Clinics[1].

If you want to reproduce our results, please keep the random seed in data_process.py so that you get same 80-10-10 split for train/dev/test.

<h1> Source files </h1>
All the files assumed that data is stored in *../dataset* directory and sub-directory strucutre of data set is maintained as

*dataset\cocacoronarycalciumandchestcts-2\Gated_release_final*
*dataset\cocacoronarycalciumandchestcts-2\Gated_release_final*

If you change the dataset directory structure, then you have to edit files. Search for "ddir" variable in these files.
    
1) **data_process.py**
  It can be used to visualize the CT scan for a patient, generate train/test/dev split. Visualization can be 3D or multiple 2D images.
  When run with "doPlot = False", it will generate files with pickle object containing PIDs of train/dev/test split. 
    gate_test_pids.dump: test PIDs
    gated_train_dev_pids.dump: train and dev PIDs
 
  To visualize, set "doPlot = True; plot3D = True; pid = <pid>"
  
  2) **train.py**: Used to train the models.
  Example:
  ```
  python train.py --train -batch_size 4 -mdir ../trained_models/unet.final.r -mname unet -loss dice -upsample_ps 10 -epochs 10
  
  See help:
  
  usage: train.py [-h] [-batch_size BATCH_SIZE] [-epochs EPOCHS] [-max_train_patients MAX_TRAIN_PATIENTS] [-dice_loss_fraction DICE_LOSS_FRACTION] [-upsample_ps UPSAMPLE_PS] [-ddir DDIR]
                [-patient_splits_dir PATIENT_SPLITS_DIR] [-mdir MDIR] [-mname MNAME] [--plot] [--train] [--hsen] [-lr LR] [-steps_per_epoch STEPS_PER_EPOCH] [-model_save_freq_steps MODEL_SAVE_FREQ_STEPS]
                [-loss {bce,dice,focal,dice_n_bce}] [--reset] [--only_use_pos_images] [--use_dev_pos_images] [--den] [-num_neg_images_per_batch NUM_NEG_IMAGES_PER_BATCH]

optional arguments:
  -h, --help            show this help message and exit
  -batch_size BATCH_SIZE
                        List of batch sizes
  -epochs EPOCHS
  -max_train_patients MAX_TRAIN_PATIENTS
                        To limit number of training examples
  -dice_loss_fraction DICE_LOSS_FRACTION
                        Total loss is sum of dice loss and cross entropy loss. This controls fraction of dice loss to consider. Set it to 1.0 to ignore class loss
  -upsample_ps UPSAMPLE_PS
                        Non zero value to enable up-sampling positive samples during training
  -ddir DDIR            Data set directory. Don't change sub-directories of the dataset
  -patient_splits_dir PATIENT_SPLITS_DIR
                        Directory in which patient splits are located.
  -mdir MDIR            Model's directory
  -mname MNAME          Model's name
  --plot                Plot the metric/loss
  --train               Train the model
  --hsen                Generate random hyper parameters
  -lr LR                List of learning rates
  -steps_per_epoch STEPS_PER_EPOCH
                        Number of steps per epoch. Set this to increase the frequency at which Tensorboard reports eval metrics. If None, it will report eval once per epoch.
  -model_save_freq_steps MODEL_SAVE_FREQ_STEPS
                        Save the model at the end of this many batches. If low,can slow down training. If none, save after each epoch.
  -loss {bce,dice,focal,dice_n_bce}
                        Pick loss from ('bce', 'dice', 'focal', 'dice_n_bce')
  --reset               To reset model
  --only_use_pos_images
                        Train with positive images only
  --use_dev_pos_images  Evaluate only on positive samples on dev set
  --den                 Enable data augmentation
  -num_neg_images_per_batch NUM_NEG_IMAGES_PER_BATCH
                        Number of positive images to be replaced with neg images per batch. Use with --only_use_pos_images

  ```
  
  2) **predict.py**: To evaluate and analyze the results
  Example: 
  ```
  python predict.py --evaluate -set test -batch_size 8 -mdir ../trained_models/unet.final.r -mname unet -loss focal

  
  usage: predict.py [-h] [-batch_size BATCH_SIZE] [-ddir DDIR] [-mdir MDIR] [-mname MNAME] [-pid PID]
                  [-loss {bce,dice,focal,dice_n_bce}] [-dice_loss_fraction DICE_LOSS_FRACTION] [--evaluate]
                  [-set {train,dev,test}] [--print_stats] [--only_use_pos_images] [-pmask_threshold PMASK_THRESHOLD]
                  [--print_agatston_score]

optional arguments:
  -h, --help            show this help message and exit
  -batch_size BATCH_SIZE
                        List of batch sizes
  -ddir DDIR            Data set directory. Don't change sub-directories of the dataset
  -mdir MDIR            Model's directory
  -mname MNAME          Model's name
  -pid PID              pid to plot
  -loss {bce,dice,focal,dice_n_bce}
                        Pick loss from ('bce', 'dice', 'focal', 'dice_n_bce')
  -dice_loss_fraction DICE_LOSS_FRACTION
                        Total loss is sum of dice loss and cross entropy loss. This controls fraction of dice loss to consider.
                        Set it to 1.0 to ignore class loss
  --evaluate            Evaluate the model
  -set {train,dev,test}
                        Specify set (train|dev|test) to evaluate or predict on
  --print_stats         Predict on all patients and print number of predicted calcified pixels
  --only_use_pos_images
                        Evaluate with positive images only
  -pmask_threshold PMASK_THRESHOLD
                        A non-zero number will filter lesion less than this area
  --print_agatston_score
                        Print Agaston score for actual and predicted

  ```
  4) **models/base_model.py** - contains methods to train, custom loss functions and metrics
     models/unet.py - unet model
     models/uneta.py - unet with attention model
   New model architectures can be easily incorporated by writing models/<model>.py
  
 5) **dataGenerator.py** - class to create input X and output Y for training and evaluating model.
  

References:
[1] (Stanford University Stanford AIMI. Coca - coronary calcium and chest cts, 2021.
data retrieved from Stanford AIMI, https://stanfordaimi.azurewebsites.net/datasets/
e8ca74dc-8dd4-4340-815a-60b41f6cb2aa)
