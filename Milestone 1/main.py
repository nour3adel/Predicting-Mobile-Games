from Preprocessing.traindata import DataPreprocessing
from Preprocessing.testdata import DataPreprocessing_Test
from Visualization.visualizing import PCAModelVisualizer
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import pickle

# /////////////////////////////////////////////////////////////////
print("####################################################################")
x = float(input("Enter 1 To Training Or 2 To Testing : "))
print("####################################################################")
if x == 1:
   #Training Data
    data_preprocess = DataPreprocessing('Data/games-regression-dataset.csv')
    data_preprocess.preprocess_all()
    X_train, y_train, X_val, y_val, X_test, y_test = data_preprocess.split_data_then_scale(
    0.20, 0.20)
    
    # ######################################################################
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    filename1 = 'model/linear.pkl'
    pickle.dump(lr, open(filename1, 'wb'))

    print('///////////////linear_regression/////////////////////////')
    print(f"Mean Square Error on test set: {mean_squared_error(y_test, y_pred)}")
    print(f"R2 Score on test set: {r2_score(y_test, y_pred)}\n")
    print("_____________________________________________________________")
    ######################################################################
    rand = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rand.fit(X_train, y_train)
    y_pred_test_rand = rand.predict(X_test)
    filename2 = 'model/RFG.pkl'
    pickle.dump(rand, open(filename2, 'wb'))

    print('//////////////Random Forest regression//////////////////////')
    print(f"Mean Square Error on test set: {mean_squared_error(y_test, y_pred_test_rand)}")
    print(f"R2 Score on test set: {r2_score(y_test, y_pred_test_rand)}\n")
    print("_____________________________________________________________")
    
    #####################################################################

    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train)
    poly_model = LinearRegression().fit(X_train_poly, y_train)
    poly_features.fit(X_train, y_train)
    pred_pol = poly_model.predict(poly_features.fit_transform(X_test))

    filename3 = 'model/poly.pkl'
    pickle.dump(poly_model, open(filename3, 'wb'))

    print('//////////////polynomial_regression//////////////////////')
    print(f"MSE on test set: {metrics.mean_squared_error(y_test, pred_pol)}")
    print(f"R2 Score on test set: {r2_score(y_test, pred_pol)}\n")
    
    
    
    
    # Create a PCAModelVisualizer object and fit_transform on the dataset
    pca_visualizer = PCAModelVisualizer(X_train)
    pca_visualizer.fit_transform(5)

    # Plot the transformed dataset with the target variable
    pca_visualizer.plot(y_train, 'PCA on train set')

    # Create a PCAModelVisualizer object and fit_transform on the dataset
    pca_visualizer = PCAModelVisualizer(X_test)
    pca_visualizer.fit_transform(5)

    # Plot the transformed dataset with the target variable
    pca_visualizer.plot(y_test, 'PCA on test set')
    
    
else:
    
    #Testing Data
    data_preprocess2 = DataPreprocessing_Test('Data/data_for_test.csv')
    data_preprocess2.preprocess_all()
    x , y = data_preprocess2.DataScaling()
    
    ######################################################################
    linear_model = pickle.load(open("model/linear.pkl", 'rb'))
    prediction = linear_model.predict(x)
    print('///////////////linear_regression/////////////////////////')
    print('Mean Square Error', metrics.mean_squared_error(y, prediction))
    result = linear_model.score(x, y)
    print('R2 Score', result)
    ######################################################################
    rand =  pickle.load(open("model/RFG.pkl", 'rb'))
    prediction = rand.predict(x)

    print('//////////////Random Forest regression//////////////////////')
    print('Mean Square Error', metrics.mean_squared_error(y, prediction))
    result = rand.score(x, y)
    print('R2 Score', result)
    #####################################################################
    poly_model = pickle.load(open("model/poly.pkl", 'rb'))
    poly_features = PolynomialFeatures(
                degree=2).fit_transform(x)
    prediction = poly_model.predict(poly_features)

    print('//////////////polynomial_regression//////////////////////')
    print('Mean Square Error', metrics.mean_squared_error(y, prediction))
    result = poly_model.score(poly_features, y)
    print('R2 Score', result)

    # Create a PCAModelVisualizer object and fit_transform on the dataset
    pca_visualizer = PCAModelVisualizer(x)
    pca_visualizer.fit_transform(5)

    # Plot the transformed dataset with the target variable
    pca_visualizer.plot(y, 'PCA on test set')