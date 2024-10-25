import numpy as np
from model_maker import model_maker

def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

def run_experiments(num_of_repeats, X_train, y_train, X_test, y_test):
    scores = list()
    for n in range(num_of_repeats):
        score = model_maker(X_train, y_train, X_test, y_test)
        score = score * 100.0
        print('>#%d: %.3f' % (n+1, score))
        scores.append(score)
    summarize_results(scores)