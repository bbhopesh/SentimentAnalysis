import sentiment_analysis.learning.tune_params_nb as tune_params_nb
import sentiment_analysis.learning.tune_params_dt as tune_params_dt
import sentiment_analysis.learning.tune_params_svm as tune_params_svm
import sentiment_analysis.learning.tune_params_svm1 as tune_params_svm1
import sentiment_analysis.learning.tune_params_svm2 as tune_params_svm2
import sentiment_analysis.learning.tune_params_lr as tune_params_lr
import sentiment_analysis.learning.tune_params_perceptron as tune_params_perceptron
import sentiment_analysis.learning.tune_params as tune_params
import sys


def main(to_run):
    if to_run == "nb":
        tune_params_nb.main()
    elif to_run == "dt":
        tune_params_dt.main()
    elif to_run == "svm":
        tune_params_svm.main()
    elif to_run == "svm1":
        tune_params_svm1.main()
    elif to_run == "svm2":
        tune_params_svm2.main()
    elif to_run == "lr":
        tune_params_lr.main()
    elif to_run == "perceptron":
        tune_params_perceptron.main()
    elif to_run == "all":
        tune_params.main()


if __name__ == "__main__":
    main(sys.argv[1])


