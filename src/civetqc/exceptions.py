class VariableNotFoundError(Exception):
    """ raised when a required variable is not found in CSV file """
    pass


class DuplicateIdentifierError(ValueError):
    """ raised when a value for the ID variable appears more than once """
    pass


class NegativeQCRatingError(ValueError):
    """ raised when negative QC value is in CSV file """
    pass


class DataFrameMergerError(Exception):
    """ raised when cannot merge dataframes on key var due to type """
    pass


class ModelNotFoundError(FileNotFoundError):
    """ raised when path to saved model does not exist """
    pass


class InvalidClassifierError(TypeError):
    """ raised when non-sklearn classifier object is passed """
    pass
