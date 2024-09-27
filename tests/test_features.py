from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # TODO: Add a test of your own below this line
    def test_standard_scaler(self):
        scaler = StandardScaler()
        data=[[0.2, 0.4], [0.2, 0.4], [0.8, 1.0], [0.8, 1.0]]
        expected1=np.array([0.5, 0.7])
        expected2=np.array([0.3, 0.3]) 
        trans=np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        assert np.allclose(scaler.mean,expected1),f"Scaler transform does not return expected values, Expected mean:{expected1}, Got {scaler.mean}"
        assert np.allclose(scaler.std,expected2),f"Scaler transform does not return expected values, Expected std:{expected2}, Got {scaler.std}"
        result=scaler.transform(data)
        assert np.allclose(result,trans),f"Scaler transform does not return expected values, Expected transform:{trans}, Got:{result}"

    def test_label_encoder(self):
        encoder=LabelEncoder()
        data=['bowl', 'basket', 'chicken', 'bowl', 'chicken']
        expected_classes=np.array(['basket', 'bowl', 'chicken'])
        trans=[1,0,2,1,2]
        encoder.fit(data)
        assert (encoder.classes_==expected_classes).all(), f"Expected classes {expected_classes},but got {encoder.classes_}"
        result=encoder.transform(data)
        assert (result==trans), f"Expected transform {trans}, but got {result}"
        fit_transform_result=encoder.fit_transform(data)
        assert fit_transform_result==trans, f"Expected fit_transform {trans}, but got {fit_transform_result}"
    
    def test_standard_scaler_withoutnumber(self):
      scaler=StandardScaler()
      data=[["c"], ["r"], ["x"]]
      try:
        scaler.fit(data)
      except ValueError as e:
        assert str(e)=="Input contains values without number.", f"Unexpected error message: {str(e)}"

if __name__ == '__main__':
    unittest.main()
