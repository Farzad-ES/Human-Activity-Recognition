from Utils.data_preprocessing import data_preprocessing
from Utils.load_data import load_X, load_y
from Utils.model_maker import model_maker, save_model_and_weights
from Utils.run_experiments import run_experiments

X_train=load_X("train", "./UCI HAR dataset/UCI HAR dataset")
X_test=load_X("test", "./UCI HAR dataset/UCI HAR dataset")
y_train, y_test=load_y("./UCI HAR dataset/")

X_train, y_train, X_test, y_test=data_preprocessing(X_train, y_train, X_test, y_test)

accuracy, conv1d_model=model_maker(X_train, y_train, X_test, y_test)
print("Accuracy(for training and running the model once): %.3f%%" % (accuracy))
save_model_and_weights(conv1d_model, "./1D CNN Model")

run_experiments(10, X_train, y_train, X_test, y_test)
