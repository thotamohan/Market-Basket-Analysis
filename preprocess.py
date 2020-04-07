import json
import sys
import pyspark
import time
from operator import add
from pyspark import SparkContext
import itertools
from itertools import chain
from collections import defaultdict
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
sc = SparkContext("local[*]")
SparkContext.setSystemProperty('spark.executor.memory', '4g')
reviews = sc.textFile('review (1).json').persist()
review_rdd=reviews.map(lambda x:json.loads(x))
business= sc.textFile('business (1).json').persist()
business_rdd=business.map(lambda x:json.loads(x))
check_rdd=business_rdd.filter(lambda x:x['state']=='NV').map(lambda x:(x['business_id'],x['state']))
ch_review=review_rdd.map(lambda x:(x['business_id'],x['user_id']))
k=ch_review.join(check_rdd)
final_rdd=k.map(lambda x:(x[1][0],x[0]))
df = sqlContext.createDataFrame(final_rdd, ['user_id', 'business_id'])
df.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save('csv_out')