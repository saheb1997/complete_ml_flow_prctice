import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 
import dagshub


dagshub.init(repo_owner='saheb1997', repo_name='complete_ml_flow_prctice', mlflow=True)
#Ml flow expect for http or htp url not file path so that it is using http://localhost:5000
# mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_tracking_uri(" https://dagshub.com/saheb1997/complete_ml_flow_prctice.mlflow/")


wine=load_wine()
X=wine.data
y=wine.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

max_depth=10
n_estimators=5

# when mention this auto log it will log every thing like parameter,metric,tag,artifact.model
mlflow.autolog()

#mention the experiment name This is use for when we have multiple experiment Even experiment name is not there it will create 
mlflow.set_experiment("Wine-classification")

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    # print("Accuracy:",accuracy)

    # mlflow.log_metric("accuracy",accuracy)
    # mlflow.log_param("max_depth",max_depth)
    # mlflow.log_param("n_estimators",n_estimators)

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm,annot=True,cmap="viridis",xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    # even after use auto log also we have to write this line of code to log artifact
    # .lmlflowog_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tag("Author", "saheb")
    mlflow.set_tag("project", "mlflow")
    mlflow.set_tag("model", "RandomForest")

   # log model with input example
    input_example = X_test[:5]
    # mlflow.sklearn.log_model(rf, "randomforest_model", input_example=input_example)
    print("accuracy:", accuracy)


    