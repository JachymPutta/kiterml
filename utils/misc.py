import os

from constants import OUTPUT_FILE, VERBOSE, TO_FILE, MULT_FACTOR, DUP_FACTOR, DATA_LOCATION


def freeze_reqs():
    get_ipython().system("pip freeze > './requirements.txt'")

def log(file = OUTPUT_FILE, s = ""):
    if VERBOSE:
        print(s)
    if TO_FILE:
        if os.path.exists(file):
            append_write = 'a'
        else:
            append_write = 'w'
        fh = open(file, append_write)
        fh.write(s + '\n')
        fh.close()

def merge_dicts(dicts):
    merged_dict = {}
    for key in dicts[0].keys():
        merged_dict[key] = []
        for d in dicts:
            merged_dict[key].append(d[key])
    return merged_dict


def write_results(file, evals):
    log(file, "Evaluation results")
    log(file, '-' * 80)
    log(file, "Data metadata:\n" +
    "  Location: " + DATA_LOCATION + "\n" +
    "  MULT_FACTOR = " + str(MULT_FACTOR) + "\n" +
    "  DUP_FACTOR = " + str(DUP_FACTOR) + "\n")
    log(file, "Percentage of data used for training:")

    s = ""
    for sz in evals['train_sz']:
        s += str(sz[0]) + "% (" + str(sz[1]) + ")  " 

    log(file, s + "\n")
    log(file, "Average Percentage Errors:" )
    log(file, ' '.join(map(str, evals['abs_error'])))
    log(file, '-' * 80)
    log(file, "\n\n")

def write_config(file, models):
    for i, model in enumerate(models):
        print(file, f"Model {i+1}")
        print(file, model.get_config())
        print(file, "####################################")
