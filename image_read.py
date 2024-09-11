import pandas as pd
from zipfile import ZipFile 

with ZipFile("dataset_zip.zip", 'r') as zObject: 
	zObject.extractall( 
		path="Dataset/") 

