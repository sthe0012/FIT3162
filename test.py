import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from glob import glob
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from Multimodal_mexp_gaze import predict_deception
import joblib


class TestOpenFaceIntegration(unittest.TestCase):
    def setUp(self):
        self.cmd_base = "/Users/jingweiong/openFace/OpenFace/build/bin/"
        self.video_list = [
            "/Users/jingweiong/Downloads/Real-life_Deception_Detection_2016/Clips/Deceptive/trial_lie_001.mp4",
            "/Users/jingweiong/Downloads/Real-life_Deception_Detection_2016/Clips/Deceptive/trial_lie_014.mp4"
        ]
        self.expected_outputs = [
            "/Users/jingweiong/Downloads/Deception_detection_output/Gaze_reallifedeception_trial_lie_001.csv",
            "/Users/jingweiong/Downloads/Deception_detection_output/Gaze_reallifedeception_trial_lie_014.csv"
        ]

    @patch('os.system')
    def test_openface_feature_extraction(self, mock_system):
        # Setup mock to simulate successful OpenFace command execution
        mock_system.return_value = 0

        gaze_trial = 0
        dict_input_output = {}
        output_filename_list = []

        for filename in self.video_list:
            output_filename = ""
            ip = filename
            if "Real-life_Deception_Detection_2016" in filename:
                x = os.path.split(filename)[0]
                x1 = os.path.split(filename)[-1]
                output_filename = "Gaze_reallifedeception_" + x1.replace(".mp4", ".csv")
                op = "/Users/jingweiong/Downloads/Deception_detection_output/" + output_filename
                cmd2 = "FeatureExtraction -f " + ip + " -of " + op + " -gaze"
                o1 = os.system(self.cmd_base + cmd2)
                # Assert that os.system returned 0 (successful execution)
                self.assertEqual(o1, 0, f"OpenFace command failed for {ip}")
                if o1 == 0:
                    gaze_trial += 1
                    dict_input_output[filename] = op
                    output_filename_list.append(op)

        # Test the counts of processed files
        self.assertEqual(gaze_trial, len(self.video_list))
        self.assertEqual(len(output_filename_list), len(self.video_list))
        # Ensure command was called correctly
        for expected_output, input_video in zip(self.expected_outputs, self.video_list):
            mock_system.assert_any_call(self.cmd_base + "FeatureExtraction -f " + input_video + " -of " + expected_output + " -gaze")

        # Check the mapping dictionary
        for video, output in dict_input_output.items():
            self.assertIn(video, self.video_list)
            self.assertTrue(output.endswith(".csv"))
            self.assertIn(output, self.expected_outputs)


class TestDirectoryFileCheck(unittest.TestCase):

    @patch('glob.glob')
    @patch('pandas.read_csv')
    def test_files_in_directory(self, mock_read_csv, mock_glob):
        # Mock the directory contents
        dir_path = "/Users/jingweiong/Downloads/Deception_detection_output_mexp_gaze"
        mock_glob.return_value = [
            f"{dir_path}/file1.csv",
            f"{dir_path}/file2.csv",
            f"{dir_path}/Annotation_mexp_features.csv"
        ]
        
        # Mock CSV reading behavior
        # Ensure there's a return for each file expected to be read
        mock_read_csv.side_effect = [
            pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}),  # For file1.csv
            pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}),         # For file2.csv
            pd.DataFrame({'annotation': [1, 0, 1]})                # For Annotation_mexp_features.csv
        ]
        
        data_shape_all = pd.DataFrame()
        data_path = {'example_key': 'example_value'}
        
        for key in data_path.keys():
            count = 0
            data_shape, file_names = [], []
            for filepath in mock_glob(f"{dir_path}/*.csv"):
                file_shape = pd.read_csv(filepath).shape
                filename = filepath.split('/')[-1]
                for reps in (("Mexp_", ""), ("Gaze_", "")):
                    filename = filename.replace(*reps)
                if filename not in ['Annotation_mexp_features.csv', 'Annotation_gaze_features.csv']:
                    data_shape.append([file_shape[0], file_shape[1]])
                    file_names.append(filename)
                    count += 1
            data_shape = pd.DataFrame(data_shape)
            data_shape.columns = [key + str(0), key + str(1)]
            data_shape.index = pd.Series(file_names)

            # Ensure index values are unique before concatenating
            data_shape = data_shape[~data_shape.index.duplicated(keep='first')]
            
            data_shape_all = pd.concat([data_shape_all, data_shape], axis=1, sort=True)
            print(f"No. of files in {key}: ", count)
        
        # Assertions for validation
        self.assertEqual(count, 2)  # Ensure 2 annotation files are processed
        self.assertIn('file1.csv', data_shape_all.index)
        self.assertIn('file2.csv', data_shape_all.index)
        self.assertEqual(data_shape_all.shape[1], 2)  # Should have 2 columns as specified


class TestSVMClassifier(unittest.TestCase):
    def setUp(self):
        # Generate a synthetic classification dataset
        num_samples = 100
        time_steps = 10
        num_features = 5
        self.X, self.y = make_classification(
            n_samples=num_samples,
            n_features=time_steps * num_features,
            n_classes=2,
            random_state=42
        )
        # Reshape to a simulated 3D shape (num_samples, time_steps, num_features)
        self.mexp_features = self.X.reshape(num_samples, time_steps, num_features)
        self.y_train = self.y
        # Create a flattened 2D array suitable for training
        self.flattened_features = self.mexp_features.reshape(num_samples, time_steps * num_features)
        # Create an SVM classifier instance
        self.svm_classifier = SVC(kernel='linear')

    def test_svm_training(self):
        # Check for NaN values in the dataset
        nan_indices = np.isnan(self.flattened_features)
        if np.any(nan_indices):
            # Handle NaN values by imputing the mean of each feature
            mean_features = np.nanmean(self.flattened_features, axis=0)
            self.flattened_features[nan_indices] = np.take(mean_features, np.nonzero(nan_indices)[1])

        # Train the SVM classifier
        self.svm_classifier.fit(self.flattened_features, self.y_train)

        # Check if the model is trained by predicting on the training data
        try:
            predictions = self.svm_classifier.predict(self.flattened_features)
            self.assertEqual(len(predictions), len(self.y_train), "Mismatch in the number of predictions and training labels.")
        except NotFittedError:
            self.fail("The SVM classifier did not fit the training data successfully.")

class TestSVMClassifierDeceptive(unittest.TestCase):
    def setUp(self):
        # Load the trained model
        self.model_path = '/Users/jingweiong/Downloads/Deception-Detection-master/multimodal_mexp_and_gaze.pkl'
        self.model = joblib.load(self.model_path)

        # Set paths to your specific deceptive CSV files
        self.gaze_csv_path = '/Users/jingweiong/Downloads/Deception_detection_output/Gaze_reallifedeception_trial_lie_005.csv'
        self.mexp_csv_path = '/Users/jingweiong/Downloads/Deception_detection_output_mexp/Mexp_reallifedeception_trial_lie_005.csv'
        self.max_columns = 576  # Adjust this to match your model's input dimensions

    def test_prediction_deceptive(self):
        # Call your predict_deception function with actual data
        prediction = predict_deception(
            self.model, 
            self.gaze_csv_path, 
            self.mexp_csv_path, 
            max_columns=self.max_columns
        )

        # Check that the result is "Deceptive"
        self.assertEqual(prediction, "Deceptive", f"Expected 'Deceptive' but got '{prediction}'")


if __name__ == '__main__':
    unittest.main()
