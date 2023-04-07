from constants import OUTPUT_FILE, VERBOSE, TO_FILE


def freeze_reqs():
    get_ipython().system("pip freeze > './requirements.txt'")

def log(s):
    if VERBOSE:
        print(s)
    if TO_FILE:
        if os.path.exists(OUTPUT_FILE):
            append_write = 'a'
        else:
            append_write = 'w'
        fh = open(OUTPUT_FILE, append_write)
        fh.write(s + '\n')
        fh.close()

def write_results(all_train_sizes, all_percentage_errors, all_evals):
    log("Evaluation results")
    log('=' * 80)
    log("Data metadata:\n" +
    "  Location: " + DATA_LOCATION +
    "\n  Size: "+ str(len(d)) + " points\n" +
    "  MULT_FACTOR = " + str(MULT_FACTOR) + "\n")
    log("Percentage of data used for training:")

    s = ""
    for sz in all_train_sizes:
        s += str(sz[0]) + "%% (" + str(sz[1]) + ")  " 

    log(s + "\n")
    log("Average Percentage Errors:" )
    log(' '.join(map(str, all_percentage_errors)))
    log("")
    log("Evaluation results:")
    log(' '.join(map(str, all_evals)))
    log('=' * 80)
    log("\n\n")
