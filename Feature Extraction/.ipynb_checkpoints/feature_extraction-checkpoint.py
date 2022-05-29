'''
Defines methods that conduct the feature extraction process with tsfresh 
and analysis of these features with PCA.
'''
import numpy as np
import pickle
import pandas as pd
from tsfresh import extract_relevant_features
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def conduct_feature_extraction(data_files, labels_arr):
    '''
    Constructs the dataframe needed to pass into tsfresh's feature extractor.
    Each row corresponds to a single star's datapoints. Runs tsfresh
    feature extractor on the data.
    
    Params:
        - data_files: array of data files containing each star's data
        - labels_arr: array of labels to use for each star, same order as data_files
    '''
    
    max_data_points_per_star = 200
    data_labels = []
    
    column_names = ['id', 'name', 'x']
    df = pd.DataFrame(columns = column_names)
    id_val = 0
    
    # Iterate through each file's data and append to dataframe
    for file, label in zip(data_files, labels_arr):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            
        for item in data:
            length = len(item[0])
            
            times = np.array(item[0])
            mags = np.array(item[1])
            
            if (length > max_data_points_per_star):
                ids = np.full(max_data_points_per_star, id_val)
            else:
                ids = np.full(length, id_val)
            
            to_add = np.array([ids, times, rvs]).T
            if (len(to_add) == 1):
                continue
        
            new_df = pd.DataFrame(to_add, columns = df.columns)
            df = pd.concat([df, new_df], ignore_index=True)
            data_labels.append(label)
            ide_val += 1
            
    df = df.astype(float)
    ids = np.array([i for i in range(0, id_val)])
    data = {'id':ids, 'y':data_labels}
    y = pd.Dataframe(data['y']).squeeze()

    # Conduct feature extraction
    features_filtered_direct = extract_relevant_features(df, ym
                                                        column_id = 'id', column_sort='time')
    
    x_data = features_filtered_direct.to_numpy()
    y_data = y.to_numpy()
    
    with open('extracted_features_200', 'wb') as f:
        pickle.dump(x_data, f)
    
    with open('../Data/data_labels', 'wb') as f:
        pickle.dump(y_data, f)


def conduct_pca(num_components):
    '''
    Conducts principal component analysis with the maximum likelihood number
    of components or some set number of components. Analyzes the magnitude
    of the eigenvalues to determine which components are the most important
    and explain the greatest variance.

    It was found that significant number of the features are needed to even get 90% explained
    variance. With 10 components, we only get 60% explained variance, with 100 components,
    we get 90% explained variance. The MLE returns most of the components.
    
    Params:
        - num_components: the number of components to conduct PCA on
    '''
    
    with open('extracted_features_200', 'wb') as f:
        x_data = pickle.load(f)
    
    with open('../Data/data_labels', 'wb') as f:
        y_data = pickle.load(f)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0)
    
    scaler = StandardScaler()
    normalized_train_X = scaler.fit_transform(x_train)
    
    pca = PCA(n_components= 'mle', svd_solver = 'full')
    pca.fit(normalized_train_X)
    components = pca.components_
    
    print('MLE PCA Fit: ' + str(components.shape[1]) + ' features chosen')
    print('Sum of Variance Ratios: ' + str(np.sum(pca.explained_variance_ratio_)))
    print('Components: ' + str(components))
    
    pca = PCA(n_components = num_components, svd_solver = 'full')
    pca.fit(normalized_train_X)
    components = pca.components_
    
    print(str(n_components) + 'PCA Fit: ')
    print('Sum of Variance Ratios: ' + str(np.sum(pca.explained_variance_ratio_)))
    print('Components: ' + str(components))


if __name__ == '__main__':
    files = ['../Data/raw_magnitude_data/irregular_data_limit', '../Data/raw_magnitude_data/eb_data_limit',
            '../Data/raw_magnitude_data/mira_data_limit', '../Data/raw_magnitude_data/rr_data_limit']
    
    labels = ['irregular_var', 'eb_var', 'mira_var', 'rr_var']
    conduct_feature_extraction(files, labels)
    conduct_pca(10)
