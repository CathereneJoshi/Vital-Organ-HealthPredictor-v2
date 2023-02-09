import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

import pickle
import numpy as np
model1 = pickle.load(open('finalized_model.sav', 'rb'))

testcases = [
    '49,0,3,160,180,0,0,156,0,1,2',
    "41,0,1,130,204,0,0,172,0,1,2"
]
for testcase in testcases:
    int_param = [int(x) for x in testcase.split(',')]
    final_param = (np.asarray(int_param)).reshape(1,-1)
    prediction = model1.predict(final_param)
    print(prediction)