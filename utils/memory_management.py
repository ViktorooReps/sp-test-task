import resource
import pickle

def limit_memory(maxsize): 
    _, hard = resource.getrlimit(resource.RLIMIT_AS) 
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, maxsize)) 

def save_obj(obj, name): 
    with open('pickled/'+ name + '.pkl', 'wb') as f:  
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) 

def load_obj(name): 
    with open('pickled/' + name + '.pkl', 'rb') as f: 
        return pickle.load(f)

