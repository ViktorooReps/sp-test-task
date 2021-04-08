class Logger:
    """Keeps track of registered fields in list_<field name>"""
    
    def __init__(self):
        self.registered = set()
        self.times_written = 0

    def register(self, *names):
        for name in names:
            self.registered.add(name)
            self.__setattr__("list_" + name, [])
            self.__setattr__("st_idx_" + name, self.times_written)

    def write(self):
        for name in self.registered:
            new_value = self.__getattribute__(name)
            self.__getattribute__("list_" + name).append(new_value)

        self.times_written += 1
