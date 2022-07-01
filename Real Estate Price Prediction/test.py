from main import *
X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set['MEDV'].copy()
X_test_fin=pipl.transform(X_test)
fin_predi=model.predict(X_test_fin)
fin_mse=mean_squared_error(Y_test,fin_predi)
fin_rmse=np.sqrt(fin_mse)
print(fin_rmse)