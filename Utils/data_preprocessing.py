from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def data_preprocessing(X_train, y_train, X_test, y_test):
    encoder=OneHotEncoder(sparse_output=False)
    y_train=encoder.fit_transform(y_train)
    y_test=encoder.fit_transform(y_test)

    sc=StandardScaler()
    X_train=sc.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test=sc.fit_transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    return X_train, y_train, X_test, y_test