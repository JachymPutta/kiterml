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

def plot_loss_curve(model_type, curves):
    plt.plot(curves[0][0], label='train')
    plt.plot(curves[0][1], label='val')

    if TO_FILE:
        plt.savefig(FIG_DIR + model_type + '_loss_curves.png')

def plot_loss_curves(model_type, curves):
    fig, axs = plt.subplots(len(TRAIN_SET_PERCENTAGE))
    
    fig.suptitle(model_type + ' Histories')
    fig.supxlabel('Epochs')
    fig.supylabel('Loss')
    
    for i in range(len(TRAIN_SET_PERCENTAGE)):
        axs[i].plot(curves[i][0], label='train')
        axs[i].plot(curves[i][1], label='val')
        axs[i].legend(['train', 'val'], loc='upper left')

    if TO_FILE:
        fig.savefig(FIG_DIR + model_type + '_loss_curves.png')
