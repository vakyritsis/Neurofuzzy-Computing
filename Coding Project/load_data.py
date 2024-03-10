from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_l1_data(df):
    target = df['category_level_1']
    features = df[['source', 'title', 'content']]
    features = features.apply(lambda row: ' '.join(row.astype(str)), axis=1)

    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target)

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

    return x_train, x_test, x_val, y_train, y_test, y_val

def load_l2_data(l2_features, l2_target):
    x_train = []
    x_test = []
    x_val = []
    y_train = []
    y_test = []
    y_val = []

    for i in range(0, 17):
        label_encoder = LabelEncoder()
        l2_target[i] = label_encoder.fit_transform(l2_target[i])

    for i in range(0, 17):
        x_train.append([])
        x_test.append([])
        x_val.append([])
        y_train.append([])
        y_test.append([])
        y_val.append([])

    for i in range(0, 17):
        x_train[i], x_test[i], y_train[i], y_test[i] = train_test_split(l2_features[i], l2_target[i], test_size=0.3, random_state=42)
        x_test[i], x_val[i], y_test[i], y_val[i] = train_test_split(x_test[i], y_test[i], test_size=0.5, random_state=42)

    return x_train, x_test, x_val, y_train, y_test, y_val