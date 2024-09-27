import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        return (x-self.minimum)/(self.maximum-self.minimum)
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def __init__(self):
        self.mean=None
        self.std=None
    def _check_is_array(self,x:np.ndarray)->np.ndarray:
        if not isinstance(x,np.ndarray):
            x=np.array(x)
        assert isinstance(x,np.ndarray),"Expected input should be an array."
        return x
    def fit(self,x:np.ndarray)->None:
        x=self._check_is_array(x)
        if not np.issubdtype(x.dtype,np.number):
          raise ValueError("Input contains values without number.")
        self.mean=np.mean(x, axis=0)
        self.std=np.std(x, axis=0)
    def transform(self,x: np.ndarray)->np.ndarray:
        x=self._check_is_array(x)
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted.")
        return (x-self.mean)/self.std
    def fit_transform(self,x:np.ndarray)->np.ndarray:
        self.fit(x)
        return self.transform(x)

class LabelEncoder:
    def __init__(self):
        self.classes_=None    
    def fit(self, x: List[str])->None:
        self.classes_=np.unique(x)
    def transform(self,x: List[str])->List[int]:
        if self.classes_ is None:
            raise ValueError("The encoder has not been fitted.")
        return [np.where(self.classes_ == label)[0][0] for label in x]
    def fit_transform(self,x: List[str])->List[int]:
        self.fit(x)
        return self.transform(x)

