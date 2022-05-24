class CivetQCError(Exception):
    """ generic class for errors with civetqc """


class MissingVariableError(CivetQCError):
    """ raised when a required variable is not found in CSV file """
    pass


class DuplicateIdentifierError(CivetQCError, ValueError):
    """ raised when a value for the ID variable appears more than once """
    pass


class NegativeQCRatingError(CivetQCError, ValueError):
    """ raised when negative QC value is in CSV file """
    pass