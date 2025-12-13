from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("CancerDiagnosis") \
    .master("local[*]") \
    .getOrCreate()

# Load your data
df = spark.read.csv("project3_data.csv", header=True, inferSchema=True)

# Basic exploration
print("=== Dataset Overview ===")
print(f"Total samples: {df.count()}")
print(f"Number of columns: {len(df.columns)}")

print("\nColumn names:")
for col in df.columns:
    print(f"  - {col}")

print("\nSchema:")
df.printSchema()

print("\nFirst 5 rows:")
df.show(5, truncate=False)

print("\nLabel distribution:")
# Try to find the label column - it might be called 'diagnosis', 'label', etc.
if 'diagnosis' in df.columns:
    df.groupBy("diagnosis").count().show()
elif 'label' in df.columns:
    df.groupBy("label").count().show()

spark.stop()