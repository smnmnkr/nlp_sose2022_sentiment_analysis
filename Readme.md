## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
# use bash make environment (LINUX/OSX only):
make run
```

## Results
```
CLASSIFIER: base                    	 ACC(train)	= 0.6368
CLASSIFIER: base                    	 ACC(eval)	= 0.6192

CLASSIFIER: stopwords               	 ACC(train)	= 0.7401
CLASSIFIER: stopwords               	 ACC(eval)	= 0.6452

CLASSIFIER: stopwords+onlyCommon1000	 ACC(train)	= 0.6617
CLASSIFIER: stopwords+onlyCommon1000	 ACC(eval)	= 0.6460

CLASSIFIER: stopwords+sharedRemoved 	 ACC(train)	= 0.9308
CLASSIFIER: stopwords+sharedRemoved 	 ACC(eval)	= 0.5156
```
