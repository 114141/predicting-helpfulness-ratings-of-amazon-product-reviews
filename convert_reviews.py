#launch new terminal in rstudio and control + shift + enter to run
#we are running this processes in python to convert to strict json for use in R

import json
import gzip

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

#name of new file to stream (new_reviews) into and old file to get from (reviews)
f = open("new_reviews_beauty.json", 'w')
for l in parse("reviews_Beauty.json.gz"):
  f.write(l + '\n')
  
