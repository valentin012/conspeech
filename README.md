#Political Speech Generator

Link to original paper: http://arxiv.org/abs/1601.03313

##Usage

1. Prepare your environment.
	
### Linux-based systems
_VirtualEnvs are highly recommended._

    $ mkvirtualenv conspeech
    $ workon conspeech
    $ pip install -r requirements.txt
   
### Windows systems
    % (contributor not familiar with dependency management on Windows -- do virtualenvs work on Win?)
    

### Running the demo script
   
    python demo.py
    python demo.py [class]               Example: python demo.py RY
    python demo.py [class] [lambda]      Example: python demo.py RY 0.25

