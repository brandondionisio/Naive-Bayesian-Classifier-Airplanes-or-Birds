import numpy as np
from sklearn.impute import SimpleImputer

# preprocess the given data
#  - all NaN values are turned into the mean for each row
#  - returns the imputed data
def preprocess(data):
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(data.T).T
    return imputed_data

# handles the likelihood distribution for feature 2 (variance)
def calculate_var(training_data):
    # separates birds and airplanes from training data (NumPy 10x600)
    birds = training_data[:10]
    airplanes = training_data[10:]

    # divides each row into buckets of 6 (NumPy 10x100x6)
    reshaped_birds = birds.reshape(birds.shape[0], -1, 6)
    reshaped_airplanes = airplanes.reshape(airplanes.shape[0], -1, 6)

    # for each bucket, calculates the variance (NumPy 10x100)
    variances_birds = np.var(reshaped_birds, axis=-1)
    variances_airplanes = np.var(reshaped_airplanes, axis=-1)
    
    # flattens all variances into likelihood distribution of size 200,
    # where each index is a bucket of 5 (NumPy 1x200)
    bins = np.arange(0, 1001, 5)
    var_hist_birds, _ = np.histogram(variances_birds.flatten(), bins=bins)
    var_hist_airplanes, _ = np.histogram(variances_airplanes.flatten(), bins=bins)

    # normalizes the distribution into probabilities (+ 0.001 to prevent 0 variances)
    likelihoods_var_birds = np.array(var_hist_birds) / 1000 + 0.001
    likelihoods_var_airplanes = np.array(var_hist_airplanes) / 1000 + 0.001
    
    return likelihoods_var_birds, likelihoods_var_airplanes

# identifies the object given the test_track and likelihood distributions
def naive_bayesian_classifier(test_track, likelihood_data, likelihoods_var_birds, likelihoods_var_airplanes):
    # sets the priors
    prior_birds = 0.5
    prior_airplanes = 0.5

    # separates birds and airplanes from likelihood_data (NumPy 1x400)
    likelihoods_speed_birds = likelihood_data[0, :]
    likelihoods_speed_airplanes = likelihood_data[1, :]

    # divides each row of test track into buckets of 6 (NumPy 1x100x6)
    reshaped_test_track = test_track.reshape(1, 100, 6)
    # for each bucket, calculates the variance (NumPy 1x100)
    var_test_track = np.var(reshaped_test_track, axis=-1)

    # initializes classifications and final classifications arrays
    classifications = []
    final_classifications = []

    for i in range(len(test_track)):
        # calculates the speed for each index in test track
        speed = round(test_track[i] * 2) / 2

        # obtains the variance for each index in test track
        var = int(var_test_track[:, i // 6] // 5)
        # if the variance is greater than 1000 (index 200), sets to highest possible variance
        if var > 199:
            var = 199

        # Feature 1: likelihood of class based on speed
        posterior_birds = prior_birds * likelihoods_speed_birds[int(speed * 2)] / (
            (likelihoods_speed_birds[int(speed * 2)] + likelihoods_speed_airplanes[int(speed * 2)] + 0.001) / (2.4))
        posterior_airplanes = prior_airplanes * likelihoods_speed_airplanes[int(speed * 2)] / (
            (likelihoods_speed_airplanes[int(speed * 2)] + likelihoods_speed_birds[int(speed * 2)] + 0.001) / (2.4))

        # Feature 2: likelihood of class based on variance
        posterior_birds = posterior_birds * likelihoods_var_birds[var] / (
            (likelihoods_var_birds[var] + likelihoods_var_airplanes[var] / 2.4))
        posterior_airplanes = posterior_airplanes * likelihoods_var_airplanes[var] / (
            (likelihoods_var_airplanes[var] + likelihoods_var_birds[var] / 2.4))
        
        # Make a decision based on the posterior probabilities
        if posterior_birds > posterior_airplanes:
            classifications.append('b')  # bird
            prior_birds = 0.9
            prior_airplanes = 0.1
        else:
            classifications.append('a')  # airplane
            prior_birds = 0.1
            prior_airplanes = 0.9

    final_classifications = 'b' if classifications.count('b') > classifications.count('a') else 'a'
    return classifications, final_classifications

def main():
    # Read data from files into NumPy arrays
    likelihood_data = np.loadtxt('likelihood.txt')
    training_data = np.loadtxt('training.txt')
    testing_data = np.loadtxt('testing.txt')

    # preprocess training and testing data
    training_data = preprocess(training_data)
    testing_data = preprocess(testing_data)

    # Feature 2: Variance for every 6s of speed (1x200)
    likelihoods_var_birds, likelihoods_var_airplanes = calculate_var(training_data)

    # Initialize classification results
    all_classifications = []
    all_final_classifications = []

    # Classify each test track
    for test_track in testing_data:
        classification, final_classification = naive_bayesian_classifier(
            test_track, likelihood_data, likelihoods_var_birds, likelihoods_var_airplanes
        )
        all_classifications.append(classification)
        all_final_classifications.append(final_classification)

    # Print results
    print("Individual Classifications:")
    for i, classification in enumerate(all_classifications):
        count_a = classification.count('a')
        count_b = classification.count('b')
        print(f"Track {i+1}: Count of 'a': {count_a}, Count of 'b': {count_b}")

    print("\nFinal Classifications:")
    for i, final_classification in enumerate(all_final_classifications):
        print(f"Track {i+1}: {final_classification}")

if __name__ == "__main__":
    main()