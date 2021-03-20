import resource

def limit_memory(maxsize): 
    _, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, maxsize)) 