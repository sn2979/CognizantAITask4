import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# number of folds for the K-fold training algorithm
K = 10

# percentage of data set for training
split = 0.75

# csv file loading function
def load_data(path: str= "path/to/csv/"):
  dataframe = pd.read_csv(f"{path}")
  dataframe = dataframe.drop(columns= ["Unnamed: 0"], inplace = True, errors= 'ignore')
  return dataframe

# function that creates target variable and predictors
def create_target_predictors(data: pd.DataFrame= None, target: str= "estimated_stock_pct"):
  if target not in data.columns
    raise Exception (f"Target: {target} is not present in the data")
  
  # predictors variable
  X = data.drop(data[target])
  # target variable
  y = data[target]

  return X, y

# machine-learning model algorithm
def training_algorithm(X: pd.DataFrame= None, y: pd.Series= None):
  accuracy = []
  for fold in range(0,K):
    # machine-learning model we're going to use to train our data
    model = RandomForestRegressor()

    # a scaler for the variables to constrain them so that the algorithm does not become greedy with large values
    scaler = StandardScaler()

    # creation of training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= split, random_state= 42)
    
    # scale down the training set
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # train the model
    trained_model = model.fit(X_train, y_train)
    
    # predict target based on trained model
    y_pred = trained_model.predict(X_test)

    # find the mean standard error (MAE) of the target predictions vs the actual data
    mae = mean_absolute_error(y_test, y_pred)
    accuracy.apend(mae)
    print(f"Fold {fold + 1} MAE = {mae:.3f}")

  # finally, print the average standard error
  print(f"AVG MAE: {sum(accuracy)/len(accuracy):.2f}")

def main(path: str= "path/to/csv/"):
  df = load_data(path)
  X, y = create_target_predictors(df, target= 'estimated_stock_pct')
  training_algorithm(X, y)
