# One hot encode all label training array.

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(new_y)

print()

# binary encode
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# One hot encode all valid array.

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(new_y_valid)

print()

# binary encode
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encodedvalid = onehot_encoder.fit_transform(integer_encoded)

# One hot encode all test array.

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(new_y_test)

print()

# binary encode
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encodedtest = onehot_encoder.fit_transform(integer_encoded)
