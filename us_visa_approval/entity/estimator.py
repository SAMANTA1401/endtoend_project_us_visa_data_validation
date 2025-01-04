import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from us_visa_approval.exception import USvisaException
from us_visa_approval.logger import logging



class TargetValueMapping:
    def __init__(self):
        self.Certified:int = 0
        self.Denied:int = 1
    def _asdict(self):
        return self.__dict__ # inbuild method
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))
    

# The __init__ method is a special method that is called when an object of the class is instantiated.
# Within the __init__ method, the class defines two instance variables: Certified and Denied, which are 
# initialized with integer values 0 and 1, respectively.
# The _asdict method returns a dictionary representation of the object's attributes.

# The self.__dict__ attribute is a dictionary that contains the object's attributes and their values.

# The reverse_mapping method returns a reversed mapping of the object's attributes.
# Within the method, the _asdict method is called to get a dictionary representation of the object's attributes.
# The zip function is used to swap the keys and values of the dictionary, effectively reversing the mapping.
# The dict function is used to convert the zipped values back into a dictionary.


# class TargetValueMapping:
#     def __init__(self):
#         self.Certified = 0
#         self.Denied = 1

#     def _asdict(self):
#         return self.__dict__
# target_feature_train_df = pd.DataFrame({
#     'target': ['Certified', 'Denied', 'Certified']
# })
# target_feature_train_df = target_feature_train_df.replace(
#     TargetValueMapping()._asdict()
# )
# print(target_feature_train_df)

#    target
#        0
#        1
#        0
    



class USvisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        logging.info("Entered predict method of UTruckModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise USvisaException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    