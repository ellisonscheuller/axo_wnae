import logging
import logging.config
from functools import reduce


class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[36;20m"
    bold_blue = "\x1b[36;1m"
    reset = "\x1b[0m"
    format = "[%(levelname)s] %(message)s"

    formats = {
        logging.INFO: blue + format + reset,
        logging.DEBUG: bold_blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger(metaclass=Singleton):
    
    def __init__(self):
        self.__logger = logging.getLogger("mainLogger")
        
        self.__main_handler = logging.StreamHandler()
        self.__main_handler.setFormatter(CustomFormatter())
        self.__logger.addHandler(self.__main_handler)

        # Create a "blank line" handler
        self.__blank_handler = logging.StreamHandler()
        self.__blank_handler.setLevel(logging.INFO)
        self.__blank_handler.setFormatter(logging.Formatter(fmt=''))
    
        self.set_level(logging.INFO)

        self.__history = []
    
    @property
    def INFO(self):
        return logging.INFO
    
    @property
    def DEBUG(self):
        return logging.DEBUG
    
    @property
    def WARNING(self):
        return logging.WARNING
    
    @property
    def ERROR(self):
        return logging.ERROR
    
    @property
    def CRITICAL(self):
        return logging.CRITICAL

    def blank_line(self):
        self.__logger.removeHandler(self.__main_handler)
        self.__logger.addHandler(self.__blank_handler)
        self.info("")
        self.__logger.removeHandler(self.__blank_handler)
        self.__logger.addHandler(self.__main_handler)

    def __send_log_message(self, level, *args, repeat=True, **kwargs):
        args = list(map(lambda x: str(x), args))
        key = level + "_" + reduce(lambda x, y: str(x) + "_" + str(y), args)
        if repeat or (not repeat and key not in self.__history):
            function = getattr(self.__logger, level)
            function(*args, *kwargs)
        if not repeat:
            self.__history.append(key)

    def info(self, *args, **kwargs):
        self.__send_log_message("info", *args, **kwargs)
        
    def debug(self, *args, **kwargs):
        self.__send_log_message("debug", *args, **kwargs)
        
    def warning(self, *args, **kwargs):
        self.__send_log_message("warning", *args, **kwargs)
        
    def error(self, *args, **kwargs):
        self.__send_log_message("error", *args, **kwargs)
        
    def critical(self, *args, **kwargs):
        self.__send_log_message("critical", *args, **kwargs)
    
    def get_level(self):
        return self.__logger.level
    
    def set_level(self, level):
        self.__logger.setLevel(level)
        self.__main_handler.setLevel(level)
    

log = Logger()
