import pandas as pd
from sklearn import metrics
import pickle as pk

career = pd.read_csv("career.csv")

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()

career["Database Fundamentals"]=label_encoder.fit_transform(career["Database Fundamentals"])
career["Computer Architecture"]=label_encoder.fit_transform(career["Computer Architecture"])
career["Distributed Computing Systems"]=label_encoder.fit_transform(career["Distributed Computing Systems"])
career["Cyber Security"]=label_encoder.fit_transform(career["Cyber Security"])
career["Networking"]=label_encoder.fit_transform(career["Networking"])
career["Software Development"]=label_encoder.fit_transform(career["Software Development"])
career["Programming Skills"]=label_encoder.fit_transform(career["Programming Skills"])
career["Project Management"]=label_encoder.fit_transform(career["Project Management"])
career["Computer Forensics Fundamentals"]=label_encoder.fit_transform(career["Computer Forensics Fundamentals"])
career["Technical Communication"]=label_encoder.fit_transform(career["Technical Communication"])
career["AI ML"]=label_encoder.fit_transform(career["AI ML"])
career["Software Engineering"]=label_encoder.fit_transform(career["Software Engineering"])
career["Business Analysis"]=label_encoder.fit_transform(career["Business Analysis"])
career["Communication skills"]=label_encoder.fit_transform(career["Communication skills"])
career["Data Science"]=label_encoder.fit_transform(career["Data Science"])
career["Troubleshooting skills"]=label_encoder.fit_transform(career["Troubleshooting skills"])
career["Graphics Designing"]=label_encoder.fit_transform(career["Graphics Designing"])
#career["Role"]=label_encoder.fit_transform(career["Role"])


y=career["Role"]
x=career.drop('Role',axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
scores = {}
knn.fit(x_train, y_train)



y_pred = knn.predict(x_test)
print('y_pred',y_pred)
#print(knn.predict([[0,3,1,2,4,1,2,0,2,4,0,2,1,0,4,5,1]]))


scores[5] = metrics.accuracy_score(y_test, y_pred)
print('Accuracy=',scores[5]*100)

with open('career_pkl.pkl','wb') as file:
    pk.dump(knn,file)