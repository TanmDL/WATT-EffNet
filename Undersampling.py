############################## Undersampling to address data imbalance ######################

# On train datasets.

from imblearn.over_sampling import RandomOverSampler  # Important library
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter  # Important library


rus = RandomUnderSampler(random_state=42)

# Resampling training datasets.
trainarray = np.reshape(new_X,(len(new_X), 224*224*3))
# Use fit_resample.
trainarray_rus,trainlabel_rus = rus.fit_resample(trainarray,new_y)

# reshaping X back to the first dims
new_X = trainarray_rus.reshape(-1,224,224,3)
new_y  = trainlabel_rus

###########################################################################################

rus = RandomUnderSampler(random_state=42)

# Resampling valid datasets.

validarray = np.reshape(new_X_valid,(len(new_X_valid), 224*224*3))
# Use fit_resample.
validarray_rus,validlabel_rus = rus.fit_resample(validarray,new_y_valid)


# reshaping X back to the first dims
new_X_valid = validarray_rus.reshape(-1,224,224,3)
new_y_valid  = validlabel_rus

###########################################################################################

# On test datasets.

rus = RandomUnderSampler(random_state=42)

# Resampling test datasets.

testarray = np.reshape(new_X_test,(len(new_X_test), 224*224*3))
# Use fit_resample.
testarray_rus,testlabel_rus = rus.fit_resample(testarray,new_y_test)

# reshaping X back to the first dims
new_X_test = testarray_rus.reshape(-1,224,224,3)
new_y_test  = testlabel_rus
