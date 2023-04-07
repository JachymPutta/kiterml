import matplotlib.pyplot as plt

from constants import OUTPUT_FILE, TRAIN_SET_PERCENTAGE, FIG_DIR, TO_FILE, VERBOSE

def plot_predictions(preds, act_vals):
    fig, axs = plt.subplots(len(TRAIN_SET_PERCENTAGE), sharex=True, sharey=True)
    
    fig.suptitle('Predictions / Actual Values')
    fig.supxlabel('True Values')
    fig.supylabel('Predictions')

    for i in range(len(TRAIN_SET_PERCENTAGE)):
        axs[i].scatter(act_vals.T, preds[i])
        axs[i].set_title(str(TRAIN_SET_PERCENTAGE[i]) + "% of data")
    
    if TO_FILE:
        fig.savefig(FIG_DIR + 'all_predictions.png')
    
def plot_errors(errors):
    fig, axs = plt.subplots(len(TRAIN_SET_PERCENTAGE), sharex=True, sharey=True)
    
    fig.suptitle('Percentage errors')
    fig.supxlabel('True Values')
    fig.supylabel('Predictions')

    for i in range(len(TRAIN_SET_PERCENTAGE)):
        axs[i].hist(errors[i].T, bins=25)
        axs[i].set_title(str(TRAIN_SET_PERCENTAGE[i]) + "% of data")
        
    if TO_FILE:
        fig.savefig(FIG_DIR + 'all_errors.png')
        
        
def plot_histories(histories):
    fig, axs = plt.subplots(len(TRAIN_SET_PERCENTAGE))
    
    fig.suptitle('Histories')
    fig.supxlabel('Accuracy')
    fig.supylabel('Loss')
    
    for i in range(len(TRAIN_SET_PERCENTAGE)):
        axs[i].plot(histories[i].history['loss'])
        axs[i].plot(histories[i].history['val_loss'])
        axs[i].legend(['train', 'val'], loc='upper left')

    if TO_FILE:
        fig.savefig(FIG_DIR + 'all_histories.png')

def plot_loss_curves(curves):
    fig, axs = plt.subplots(len(TRAIN_SET_PERCENTAGE))
    
    fig.suptitle('Histories')
    fig.supxlabel('Accuracy')
    fig.supylabel('Loss')
    
    for i in range(len(TRAIN_SET_PERCENTAGE)):
        axs[i].plot(curves[i])
        axs[i].legend(['train', 'val'], loc='upper left')

    if TO_FILE:
        fig.savefig(FIG_DIR + 'all_histories.png')
