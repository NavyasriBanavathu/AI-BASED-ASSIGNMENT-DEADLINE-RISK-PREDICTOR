
import pickle, numpy as np
model = pickle.load(open("risk_model.pkl","rb"))
def predict_risk(avg_late_days,gpa,absences):
    return int(model.predict(np.array([[avg_late_days,gpa,absences]]))[0])
