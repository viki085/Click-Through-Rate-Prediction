import sys  
from src.logger import logging

def exception_handler(error, error_detail:sys):
    """
    This function is used to handle exceptions and print the error message.
    :param error: The error message
    :param err_detail: The error details
    :return: None
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {file_name} at line number: {line_number} with error message: {error}"
    return error_message


class CustomException(Exception):
    """
    Custom exception class to handle exceptions.
    """
    def __init__(self, error_message, error_detail:sys):
        """
        Constructor to initialize the error message and error details.
        :param error_message: The error message
        :param error_detail: The error details
        """
        super().__init__(error_message)
        self.error_message = exception_handler(error_message, error_detail=error_detail)


    def __str__(self):
        return f"Error occurred: {self.error_message}"